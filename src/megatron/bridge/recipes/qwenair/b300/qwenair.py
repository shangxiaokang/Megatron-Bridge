# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""B300 recipes for QwenAir EP training with mock or indexed text data."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from megatron.core.models.qwenair import QwenAirTextConfig, estimate_qwenair_training_memory
from megatron.core.models.qwenair.ple import qwenair_ngram_metadata

from megatron.bridge.data import DirectHFSFTDatasetConfig, HFDatasetSourceConfig
from megatron.bridge.data.sft_processing import ChatSFTPreprocessingConfig
from megatron.bridge.models.qwenair import QwenAirModelProvider, QwenAirMultimodalModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_QWENAIR_DDP_BUCKET_SIZE = 40_000_000
_QWENAIR_VOCAB_SIZE = 248_320
_QWENAIR_EOD_ID = 248_044
_QWENAIR_REAL_DATA_TRAIN_ITERS = 100
_QWENAIR_IMAGE_TOKEN_ID = 248_056
_QWENAIR_VIDEO_TOKEN_ID = 248_057
_QWENAIR_VISION_START_TOKEN_ID = 248_053


def _tiny_text_config() -> dict[str, Any]:
    return {
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        # Keep the checkpoint-compatible token contract even though the model
        # geometry is intentionally tiny. This lets the same recipe consume
        # indexed data produced with the QwenAir tokenizer.
        "vocab_size": _QWENAIR_VOCAB_SIZE,
        "eos_token_id": _QWENAIR_EOD_ID,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "num_experts": 64,
        "num_experts_per_tok": 2,
        "hc_count": 2,
        "hc_lowrank": 8,
        "ple_layer_ids": [2],
        "ple_embed_dim": 32,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 257,
        "make_ngram_vocab_size_divisible_by": 16,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 8,
        # The QSA indexer uses a hard top-k and its separate training objective
        # is not part of the public QwenAir contract.  Keep short convergence
        # runs dense-equivalent at the maximum sequence length so a frozen,
        # randomly initialized selector cannot hide language-model learning.
        "indexer_budget": 128,
        "indexer_compress_ratio": 4,
        "max_reference_sequence_length": 128,
        "rope_parameters": {
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    }


def _target_text_config() -> dict[str, Any]:
    """Return the built-in 32-B300 target text geometry in HF form."""
    text = asdict(QwenAirTextConfig())
    text["model_type"] = "qwen4_exp_text"
    text["dtype"] = "bfloat16"
    text["rope_parameters"] = {
        "rope_theta": text.pop("rope_theta"),
        "partial_rotary_factor": text.pop("partial_rotary_factor"),
        "mrope_section": list(text.pop("mrope_section")),
    }
    return text


def _tiny_vision_config() -> dict[str, Any]:
    """Return a compute-bounded ViT preserving QwenAir's patch/token contract."""
    return {
        "model_type": "qwen4_exp",
        "depth": 2,
        "hidden_size": 64,
        "num_heads": 4,
        "intermediate_size": 256,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "in_channels": 3,
        "num_position_embeddings": 2304,
        "out_hidden_size": 32,
        "deepstack_visual_indexes": [],
    }


def _read_target_text_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, Mapping):
        raise TypeError("QwenAir config JSON must contain an object")
    text = payload.get("text_config", payload)
    if not isinstance(text, Mapping):
        raise TypeError("QwenAir text_config must contain an object")
    text = deepcopy(dict(text))
    if text.get("model_type") != "qwen4_exp_text":
        raise ValueError("QwenAir recipe requires model_type=qwen4_exp_text")
    return text


def _base_recipe(
    text_config: Mapping[str, Any],
    *,
    world_size: int,
    expert_model_parallel_size: int,
    seq_length: int,
    train_iters: int,
    vision_config: Mapping[str, Any] | None = None,
    audit_visual_gradient: bool = False,
) -> ConfigContainer:
    if seq_length < 2 or seq_length > int(text_config["max_position_embeddings"]):
        raise ValueError("seq_length must be in [2, max_position_embeddings]")
    if world_size % expert_model_parallel_size:
        raise ValueError("world_size must divide evenly by expert_model_parallel_size")
    vocab_size = int(text_config["vocab_size"])
    eos_token_id = int(text_config["eos_token_id"])
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id must be in [0, vocab_size)")

    text = deepcopy(dict(text_config))
    text["expert_model_parallel_size"] = expert_model_parallel_size
    planning_config = QwenAirTextConfig.from_hf_dict(text)

    # The estimator validates the same per-rank allocation guards as model
    # construction. Size those guards from the requested geometry before asking
    # it for the complete model/optimizer estimate.
    head_width = planning_config.ple_embed_dim // ((planning_config.ngram_size - 1) * planning_config.heads_per_ngram)
    ple_elements_per_rank = sum(
        qwenair_ngram_metadata(planning_config, ple_index)[3] // expert_model_parallel_size * head_width
        for ple_index in range(len(planning_config.ple_layer_ids))
    )
    routed_expert_parameters_per_rank = (
        planning_config.num_hidden_layers
        * (planning_config.num_experts // expert_model_parallel_size)
        * 3
        * planning_config.moe_intermediate_size
        * planning_config.hidden_size
    )
    planning_config.max_single_rank_ple_elements = max(1, ple_elements_per_rank)
    planning_config.max_single_rank_parameters = max(1, routed_expert_parameters_per_rank)
    estimate = estimate_qwenair_training_memory(planning_config, world_size)
    # These are explicit allocation guards. Set them to the planned local
    # shards, rather than disabling the model's fail-closed resource checks.
    text["max_single_rank_ple_elements"] = max(1, estimate.ple_parameters_per_rank)
    text["max_single_rank_parameters"] = max(1, estimate.routed_expert_parameters_per_rank)

    cfg = _pretrain_common()
    if vision_config is None:
        cfg.model = QwenAirModelProvider.from_hf_config(text, qsa_backend="te_triton")
    else:
        cfg.model = QwenAirMultimodalModelProvider.from_hf_config(
            {
                "model_type": "qwen4_exp",
                "text_config": text,
                "vision_config": deepcopy(dict(vision_config)),
                "image_token_id": _QWENAIR_IMAGE_TOKEN_ID,
                "video_token_id": _QWENAIR_VIDEO_TOKEN_ID,
                "vision_start_token_id": _QWENAIR_VISION_START_TOKEN_ID,
            },
            qsa_backend="te_triton",
            audit_visual_gradient=audit_visual_gradient,
        )
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = seq_length
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.calculate_per_token_loss = True
    cfg.model.mtp_enabled = False
    cfg.model.mtp_num_layers = 0

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = vocab_size
    cfg.tokenizer.null_tokenizer_eod_id = eos_token_id
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.dataset.seq_length = seq_length
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 1
    cfg.dataset.skip_getting_attention_mask_from_dataset = True

    cfg.train.train_iters = train_iters
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = world_size
    cfg.validation.eval_interval = train_iters + 1
    cfg.validation.eval_iters = 0

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    # Keep synchronous collectives below the NCCL element-count boundary for
    # the target expert/PLE buffer. Individual parameters remain indivisible.
    cfg.ddp.bucket_size = _QWENAIR_DDP_BUCKET_SIZE
    cfg.checkpoint.save_interval = max(1, train_iters)
    cfg.logger.log_interval = 1
    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}
    return cfg


def qwenair_tiny_pretrain_8gpu_b300_bf16_config() -> ConfigContainer:
    """Return the EP4 x EDP2 eight-B300 integration recipe."""
    return _base_recipe(
        _tiny_text_config(),
        world_size=8,
        expert_model_parallel_size=4,
        seq_length=64,
        train_iters=2,
    )


def qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config(
    *,
    dataset_revision: str,
    processor_revision: str,
    train_iters: int = 128,
    global_batch_size: int = 128,
    image_size: int = 224,
) -> ConfigContainer:
    """Return the reproducible Flickr8k image-caption pipeline-learning recipe."""
    if not dataset_revision.strip() or not processor_revision.strip():
        raise ValueError("dataset_revision and processor_revision must be immutable revisions")
    if train_iters < 1 or global_batch_size < 1:
        raise ValueError("train_iters and global_batch_size must be positive")
    if image_size < 32 or image_size % 32:
        raise ValueError("image_size must be at least 32 and divisible by patch_size * merge_size (32)")

    cfg = _base_recipe(
        _tiny_text_config(),
        world_size=8,
        expert_model_parallel_size=4,
        seq_length=128,
        train_iters=train_iters,
        vision_config=_tiny_vision_config(),
        audit_visual_gradient=True,
    )
    cfg.dataset = DirectHFSFTDatasetConfig(
        seq_length=128,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
        hf_processor_path="Qwen/Qwen3.5-0.8B",
        hf_processor_kwargs={"revision": processor_revision},
        source=HFDatasetSourceConfig(
            dataset_name="flickr8k",
            load_kwargs={"revision": dataset_revision},
        ),
        # A one-source blend preserves every image-caption pair while applying
        # the deterministic blend shuffle before the 16,384-sample run is cut.
        source_weights=[1.0],
        blend_seed=1234,
        num_workers=2,
        dataloader_type="cyclic",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=True,
        do_validation=False,
        do_test=False,
        skip_getting_attention_mask_from_dataset=False,
        pad_to_max_length=False,
        pad_to_multiple_of=1,
        enable_in_batch_packing=False,
        min_pixels=image_size * image_size,
        max_pixels=image_size * image_size,
    )
    cfg.train.global_batch_size = global_batch_size
    cfg.optimizer.lr = 1.0e-3
    # This is a 128-step integration/convergence check, not a complete
    # pretraining schedule.  Decaying immediately after warmup used only 57%
    # of the update budget of a constant 1e-3 schedule and made the last
    # quarter appear to plateau.  Hold the peak rate after warmup.
    cfg.optimizer.min_lr = cfg.optimizer.lr
    cfg.scheduler.lr_decay_style = "constant"
    cfg.scheduler.lr_warmup_iters = min(12, max(1, train_iters // 10))
    cfg.scheduler.lr_decay_iters = train_iters
    cfg.scheduler.lr_wsd_decay_iters = None
    cfg.validation.eval_interval = train_iters + 1
    cfg.validation.eval_iters = 0
    cfg.checkpoint.save_interval = train_iters
    return cfg


def configure_qwenair_indexed_data(
    config: ConfigContainer,
    data_path: str | Path,
    *,
    train_iters: int = _QWENAIR_REAL_DATA_TRAIN_ITERS,
    tokenizer_vocab_size: int | None = None,
    tokenizer_eod_id: int | None = None,
    learning_rate: float | None = None,
    min_learning_rate: float | None = None,
) -> ConfigContainer:
    """Configure a QwenAir recipe for a bounded indexed-data training run.

    ``data_path`` is the Megatron indexed-dataset prefix, without ``.bin`` or
    ``.idx``. The data must be pre-tokenized with the tokenizer matching the
    recipe's vocabulary and EOD token.

    Args:
        config: QwenAir recipe to update.
        data_path: Megatron indexed-dataset prefix.
        train_iters: Number of optimizer steps to run.
        tokenizer_vocab_size: Optional from-scratch tokenizer vocabulary size.
        tokenizer_eod_id: EOD token paired with ``tokenizer_vocab_size``.
        learning_rate: Optional peak learning rate for a bounded run.
        min_learning_rate: Minimum rate paired with ``learning_rate``.

    Returns:
        The updated recipe config.

    Raises:
        ValueError: If the path is empty or ``train_iters`` is not positive.
    """
    normalized_data_path = str(data_path).strip()
    if not normalized_data_path:
        raise ValueError("data_path must be a non-empty indexed-dataset prefix")
    if train_iters < 1:
        raise ValueError("train_iters must be positive")
    if (tokenizer_vocab_size is None) != (tokenizer_eod_id is None):
        raise ValueError("tokenizer_vocab_size and tokenizer_eod_id must be provided together")
    if (learning_rate is None) != (min_learning_rate is None):
        raise ValueError("learning_rate and min_learning_rate must be provided together")
    if learning_rate is not None and min_learning_rate is not None:
        if learning_rate <= 0 or not 0 <= min_learning_rate <= learning_rate:
            raise ValueError("learning rates must satisfy 0 <= min_learning_rate <= learning_rate")
        config.optimizer.lr = learning_rate
        config.optimizer.min_lr = min_learning_rate
    if tokenizer_vocab_size is not None and tokenizer_eod_id is not None:
        if tokenizer_vocab_size < 1 or not 0 <= tokenizer_eod_id < tokenizer_vocab_size:
            raise ValueError("tokenizer_eod_id must be in [0, tokenizer_vocab_size)")
        # This override is intended for from-scratch convergence tests with an
        # existing offline tokenizer. Update every source read by the provider
        # before model construction so the token contract cannot diverge.
        config.model.qwenair_text_config["vocab_size"] = tokenizer_vocab_size
        config.model.qwenair_text_config["eos_token_id"] = tokenizer_eod_id
        config.model.vocab_size = tokenizer_vocab_size
        config.tokenizer.vocab_size = tokenizer_vocab_size
        config.tokenizer.null_tokenizer_eod_id = tokenizer_eod_id

    # A list keeps a valid prefix containing whitespace as one dataset. The
    # config finalizer intentionally treats a string as CLI-style whitespace-
    # separated paths.
    config.dataset.data_path = [normalized_data_path]
    config.dataset.blend = None
    config.dataset.blend_per_split = None
    config.train.train_iters = train_iters
    config.validation.eval_interval = train_iters + 1
    config.validation.eval_iters = 0
    config.checkpoint.save_interval = train_iters

    # The general pretraining default has a 500-step warmup. For a bounded
    # convergence run that would consume almost the entire schedule, so use a
    # ten-percent warmup capped at ten steps and decay across the full run.
    config.scheduler.lr_warmup_iters = min(10, train_iters // 10)
    config.scheduler.lr_decay_iters = train_iters
    config.scheduler.lr_wsd_decay_iters = None
    return config


def qwenair_text_pretrain_32gpu_b300_bf16_config(
    config_path: str | Path | None = None,
    *,
    seq_length: int = 64,
) -> ConfigContainer:
    """Return a 32-B300 EP32 target-text bring-up recipe.

    With ``config_path``, the recipe reads the supplied, pinned QwenAir JSON and
    otherwise uses the built-in target text geometry. Both variants deliberately
    disable the undefined MTP objective. The conservative default is 64 tokens
    because the reference GDN recurrence retains an FP32 state for every token;
    262K context requires the unfinished chunked GDN, CP, and native
    sparse-attention work.
    """
    text_config = _target_text_config() if config_path is None else _read_target_text_config(config_path)

    return _base_recipe(
        text_config,
        world_size=32,
        expert_model_parallel_size=32,
        seq_length=seq_length,
        train_iters=10,
    )


__all__ = [
    "configure_qwenair_indexed_data",
    "qwenair_text_pretrain_32gpu_b300_bf16_config",
    "qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config",
    "qwenair_tiny_pretrain_8gpu_b300_bf16_config",
]
