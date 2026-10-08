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

"""B300 recipes for QwenAir EP training with mock text data."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from megatron.core.models.qwenair import QwenAirTextConfig, estimate_qwenair_training_memory

from megatron.bridge.models.qwenair import QwenAirModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


def _tiny_text_config() -> dict[str, Any]:
    return {
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        "vocab_size": 128,
        "eos_token_id": 4,
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
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "max_reference_sequence_length": 128,
        "rope_parameters": {
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
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
) -> ConfigContainer:
    if seq_length < 2 or seq_length > int(text_config["max_position_embeddings"]):
        raise ValueError("seq_length must be in [2, max_position_embeddings]")
    if world_size % expert_model_parallel_size:
        raise ValueError("world_size must divide evenly by expert_model_parallel_size")

    text = deepcopy(dict(text_config))
    text["expert_model_parallel_size"] = expert_model_parallel_size
    planning_config = QwenAirTextConfig.from_hf_dict(text)
    estimate = estimate_qwenair_training_memory(planning_config, world_size)
    # These are explicit allocation guards. Set them to the planned local
    # shards, rather than disabling the model's fail-closed resource checks.
    text["max_single_rank_ple_elements"] = max(1, estimate.ple_parameters_per_rank)
    text["max_single_rank_parameters"] = max(1, estimate.routed_expert_parameters_per_rank)

    cfg = _pretrain_common()
    cfg.model = QwenAirModelProvider.from_hf_config(text, qsa_backend="te_triton")
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
    cfg.tokenizer.vocab_size = int(text["vocab_size"])
    cfg.tokenizer.null_tokenizer_eod_id = int(text["eos_token_id"])
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
    cfg.checkpoint.save_interval = max(1, train_iters)
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


def qwenair_text_pretrain_32gpu_b300_bf16_config(
    config_path: str | Path,
    *,
    seq_length: int = 4096,
) -> ConfigContainer:
    """Return a 32-B300 EP32 target-text bring-up recipe.

    The recipe reads the supplied, pinned QwenAir JSON and deliberately disables
    the undefined MTP objective. It is a short-context LM bring-up configuration;
    262K context requires the unfinished CP and native sparse-attention work.
    """
    return _base_recipe(
        _read_target_text_config(config_path),
        world_size=32,
        expert_model_parallel_size=32,
        seq_length=seq_length,
        train_iters=10,
    )


__all__ = [
    "qwenair_text_pretrain_32gpu_b300_bf16_config",
    "qwenair_tiny_pretrain_8gpu_b300_bf16_config",
]
