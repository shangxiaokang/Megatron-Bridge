# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QwenAir B300 recipe contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from examples.models.qwenair.finetune_qwenair_multimodal import _apply_run_overrides, _configure_run_outputs
from examples.models.qwenair.finetune_qwenair_target_multimodal import (
    _validate_processor_contract,
    build_config,
    parse_args,
)
from megatron.core.models.qwenair import QwenAirTextConfig, estimate_qwenair_training_memory

from megatron.bridge.models.qwenair import QwenAirModelProvider, QwenAirMultimodalModelProvider, qwenair_provider
from megatron.bridge.recipes.qwenair.b300 import qwenair_target_multimodal_finetune_32gpu_b300_bf16_config
from megatron.bridge.recipes.qwenair.b300.qwenair import (
    _target_text_config,
    _tiny_text_config,
    configure_qwenair_indexed_data,
    qwenair_text_pretrain_32gpu_b300_bf16_config,
    qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config,
    qwenair_tiny_pretrain_8gpu_b300_bf16_config,
)


def _canonical_multimodal_payload() -> dict:
    text = _target_text_config()
    text.update(
        {
            "bos_token_id": 248_044,
            "layer_types": [
                "linear_attention" if (layer_index + 1) % 4 else "full_attention" for layer_index in range(48)
            ],
            "mamba_ssm_dtype": "float32",
            "mtp": {
                "hybrid": True,
                "layer_types": ["full_attention"],
                "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 1,
                "rope_theta": 10_000_000,
            },
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "output_router_logits": False,
            "pad_token_id": None,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
            "split_ngram_parts": 128,
            "use_cache": True,
        }
    )
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 248_056,
        "language_model_only": False,
        "model_type": "qwen4_exp",
        "text_config": text,
        "tie_word_embeddings": False,
        "video_token_id": 248_057,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "model_type": "qwen4_exp",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248_054,
        "vision_start_token_id": 248_053,
    }


def _write_canonical_multimodal_config(tmp_path: Path) -> Path:
    path = tmp_path / "bf16-model-config.json"
    path.write_text(json.dumps(_canonical_multimodal_payload()), encoding="utf-8")
    return path


def test_tiny_recipe_uses_ep4_edp2_compatible_policy() -> None:
    cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()

    assert isinstance(cfg.model, QwenAirModelProvider)
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.qsa_backend == "te_triton"
    assert cfg.model.qwenair_text_config["indexer_compress_ratio"] == 4
    assert cfg.model.qwenair_text_config["indexer_budget"] == 128
    assert cfg.model.mtp_num_layers == 0
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.train.global_batch_size == 8
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seq_length == 64
    assert cfg.model.vocab_size == 248_320
    assert cfg.tokenizer.vocab_size == 248_320
    assert cfg.tokenizer.null_tokenizer_eod_id == 248_044
    assert cfg.tokenizer.null_tokenizer_eod_id == _tiny_text_config()["eos_token_id"]
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.ddp.bucket_size == 40_000_000
    assert cfg.logger.log_interval == 1
    assert cfg.mixed_precision.bf16 is True
    assert cfg.optimizer.main_params_dtype == torch.float32


def test_tiny_multimodal_recipe_uses_real_shuffled_flickr8k_contract() -> None:
    cfg = qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config(
        dataset_revision="dataset-commit",
        processor_revision="processor-commit",
    )

    assert isinstance(cfg.model, QwenAirMultimodalModelProvider)
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.image_token_id == 248_056
    assert cfg.model.qwenair_vision_config["patch_size"] == 16
    assert cfg.model.qwenair_vision_config["temporal_patch_size"] == 2
    assert cfg.model.qwenair_vision_config["spatial_merge_size"] == 2
    assert cfg.dataset.source.dataset_name == "flickr8k"
    assert cfg.dataset.source.split == "train"
    assert cfg.dataset.source.load_kwargs == {"revision": "dataset-commit"}
    assert cfg.dataset.source_weights == [1.0]
    assert cfg.dataset.hf_processor_kwargs == {"revision": "processor-commit"}
    assert cfg.dataset.min_pixels == 224 * 224
    assert cfg.dataset.max_pixels == 224 * 224
    assert cfg.train.train_iters == 128
    assert cfg.train.global_batch_size == 128
    assert cfg.scheduler.lr_warmup_iters == 12
    assert cfg.scheduler.lr_decay_style == "constant"
    assert cfg.optimizer.lr == 1.0e-3
    assert cfg.optimizer.min_lr == 1.0e-3


def _multimodal_override_args(**overrides) -> SimpleNamespace:
    values = {
        "dataset_split": "train",
        "indexer_budget": None,
        "allow_untrained_sparse_indexer": False,
        "lr_decay_style": None,
        "min_lr": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_multimodal_cli_cosine_override_restores_a_real_decay_range() -> None:
    cfg = qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config(
        dataset_revision="dataset-commit",
        processor_revision="processor-commit",
    )

    _apply_run_overrides(cfg, _multimodal_override_args(lr_decay_style="cosine"))

    assert cfg.scheduler.lr_decay_style == "cosine"
    assert cfg.optimizer.lr == 1.0e-3
    assert cfg.optimizer.min_lr == 1.0e-4


def test_multimodal_cli_rejects_a_decaying_min_lr_with_constant_style() -> None:
    cfg = qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config(
        dataset_revision="dataset-commit",
        processor_revision="processor-commit",
    )

    with pytest.raises(ValueError, match="constant learning rate"):
        _apply_run_overrides(cfg, _multimodal_override_args(min_lr=1.0e-4))


def test_multimodal_cli_requires_explicit_opt_in_for_an_untrained_sparse_indexer() -> None:
    cfg = qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config(
        dataset_revision="dataset-commit",
        processor_revision="processor-commit",
    )

    with pytest.raises(ValueError, match="allow-untrained-sparse-indexer"):
        _apply_run_overrides(cfg, _multimodal_override_args(indexer_budget=8))

    _apply_run_overrides(
        cfg,
        _multimodal_override_args(indexer_budget=8, allow_untrained_sparse_indexer=True),
    )
    assert cfg.model.qwenair_text_config["indexer_budget"] == 8


def test_multimodal_cli_disables_implicit_persistent_outputs(tmp_path) -> None:
    cfg = qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config(
        dataset_revision="dataset-commit",
        processor_revision="processor-commit",
    )
    assert cfg.checkpoint.save is not None
    assert cfg.logger.tensorboard_dir is not None

    _configure_run_outputs(cfg, checkpoint_dir=None, tensorboard_dir=None)

    assert cfg.checkpoint.save is None
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.save_interval == 0
    assert cfg.logger.tensorboard_dir is None

    checkpoint_dir = tmp_path / "checkpoints"
    tensorboard_dir = tmp_path / "tensorboard"
    _configure_run_outputs(
        cfg,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
    )
    assert cfg.checkpoint.save == str(checkpoint_dir)
    assert cfg.checkpoint.load == str(checkpoint_dir)
    assert cfg.checkpoint.save_interval == cfg.train.train_iters
    assert cfg.logger.tensorboard_dir == str(tensorboard_dir)


def test_tiny_real_data_recipe_uses_indexed_data_and_bounded_schedule(tmp_path) -> None:
    data_path = tmp_path / "wikitext-qwenair_text_document"
    cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()

    configure_qwenair_indexed_data(
        cfg,
        data_path,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-4,
    )

    assert cfg.dataset.data_path == [str(data_path)]
    assert cfg.dataset.blend is None
    assert cfg.dataset.blend_per_split is None
    assert cfg.train.train_iters == 100
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100
    assert cfg.validation.eval_iters == 0
    assert cfg.optimizer.lr == 1.0e-3
    assert cfg.optimizer.min_lr == 1.0e-4
    assert cfg.model.qwenair_text_config["vocab_size"] == cfg.model.vocab_size
    assert cfg.model.qwenair_text_config["eos_token_id"] == cfg.tokenizer.null_tokenizer_eod_id

    cfg.dataset.finalize()

    assert cfg.dataset.mock is False
    assert cfg.dataset.blend == ([str(data_path)], None)


def test_real_data_tokenizer_override_updates_all_model_sources(tmp_path) -> None:
    cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()

    configure_qwenair_indexed_data(
        cfg,
        tmp_path / "openwebtext-gpt2_text_document",
        tokenizer_vocab_size=50_257,
        tokenizer_eod_id=50_256,
    )

    assert cfg.model.qwenair_text_config["vocab_size"] == 50_257
    assert cfg.model.qwenair_text_config["eos_token_id"] == 50_256
    assert cfg.model.vocab_size == 50_257
    assert cfg.tokenizer.vocab_size == 50_257
    assert cfg.tokenizer.null_tokenizer_eod_id == 50_256


def test_real_data_tokenizer_override_requires_a_complete_contract(tmp_path) -> None:
    cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()

    with pytest.raises(ValueError, match="provided together"):
        configure_qwenair_indexed_data(
            cfg,
            tmp_path / "data_text_document",
            tokenizer_vocab_size=50_257,
        )


def test_real_data_learning_rate_override_requires_a_complete_contract(tmp_path) -> None:
    cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()

    with pytest.raises(ValueError, match="provided together"):
        configure_qwenair_indexed_data(
            cfg,
            tmp_path / "data_text_document",
            learning_rate=1.0e-3,
        )


def test_real_data_prefix_with_whitespace_is_not_split(tmp_path) -> None:
    data_path = tmp_path / "real text" / "data_text_document"
    cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()

    configure_qwenair_indexed_data(cfg, data_path)
    cfg.dataset.finalize()

    assert cfg.dataset.blend == ([str(data_path)], None)


def test_tiny_recipe_config_constructs_te_triton_model(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()
    single_rank_text = dict(cfg.model.qwenair_text_config)
    single_rank_text["expert_model_parallel_size"] = 1
    single_rank_text["max_single_rank_ple_elements"] *= cfg.model.expert_model_parallel_size
    single_rank_text["max_single_rank_parameters"] *= cfg.model.expert_model_parallel_size

    def qsa_triton(
        query,
        key,
        value,
        selected_key_blocks,
        *,
        scale=None,
        validate_indices=True,
    ):
        return query

    monkeypatch.setattr(
        qwenair_provider.importlib,
        "import_module",
        lambda _name: SimpleNamespace(qsa_triton_attention=qsa_triton),
    )
    provider = QwenAirModelProvider.from_hf_config(
        single_rank_text,
        qsa_backend=cfg.model.qsa_backend,
    )

    model = provider.provide()

    assert model.config.indexer_compress_ratio == 4
    assert model.model.layers[3].self_attn.backend == "te_triton"


def test_target_multimodal_recipe_matches_canonical_32_b300_run(tmp_path: Path) -> None:
    config_path = _write_canonical_multimodal_config(tmp_path)

    cfg = qwenair_target_multimodal_finetune_32gpu_b300_bf16_config(
        config_path,
        dataset_revision="dataset-commit",
        processor_revision="processor-commit",
    )

    assert isinstance(cfg.model, QwenAirMultimodalModelProvider)
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.seq_length == 128
    assert cfg.model.num_layers == 48
    assert cfg.model.hidden_size == 2560
    assert cfg.model.num_moe_experts == 512
    assert cfg.model.qwenair_text_config["mamba_ssm_dtype"] == "float32"
    assert cfg.model.qwenair_text_config["require_fused_gdn"] is True
    assert cfg.model.qwenair_text_config["moe_expert_backend"] == "loop"
    assert cfg.model.qwenair_text_config["split_ngram_parts"] == 128
    assert cfg.model.qwenair_text_config["indexer_budget"] == 2048
    assert cfg.model.qwenair_text_config["mtp_num_hidden_layers"] == 1
    assert cfg.model.mtp_enabled is False
    assert cfg.model.mtp_num_layers == 0
    assert cfg.model.qwenair_vision_config["depth"] == 27
    assert cfg.model.qwenair_vision_config["hidden_size"] == 1152
    assert cfg.model.qwenair_vision_config["intermediate_size"] == 4304
    assert cfg.model.qwenair_vision_config["out_hidden_size"] == 2560
    assert cfg.model.image_token_id == 248_056
    assert cfg.model.video_token_id == 248_057
    assert cfg.model.vision_start_token_id == 248_053
    assert cfg.dataset.source.dataset_name == "flickr8k"
    assert cfg.dataset.source.split == "train"
    assert cfg.dataset.source.load_kwargs == {"revision": "dataset-commit"}
    assert cfg.dataset.hf_processor_kwargs == {"revision": "processor-commit"}
    assert cfg.dataset.seq_length == 128
    assert cfg.dataset.min_pixels == 224 * 224
    assert cfg.dataset.max_pixels == 224 * 224
    assert cfg.train.train_iters == 1024
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size == 128
    assert cfg.optimizer.lr == 3.0e-4
    assert cfg.optimizer.min_lr == 3.0e-5
    assert cfg.scheduler.lr_decay_style == "cosine"
    assert cfg.scheduler.lr_warmup_iters == 64
    assert cfg.scheduler.lr_decay_iters == 1024
    assert cfg.checkpoint.save is None
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.save_interval == 0
    assert cfg.logger.tensorboard_dir is None


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("model_type", "qwen3_8_flash_next", "model_type=qwen4_exp"),
        ("language_model_only", True, "language_model_only=false"),
        ("image_token_id", 248_055, "multimodal token contract"),
    ],
)
def test_target_multimodal_recipe_rejects_noncanonical_wrapper(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    payload = _canonical_multimodal_payload()
    payload[field] = invalid_value
    config_path = tmp_path / "invalid-config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        qwenair_target_multimodal_finetune_32gpu_b300_bf16_config(
            config_path,
            dataset_revision="dataset-commit",
            processor_revision="processor-commit",
        )


def test_target_multimodal_cli_defaults_to_no_checkpoint(tmp_path: Path) -> None:
    config_path = _write_canonical_multimodal_config(tmp_path)
    args = parse_args(["--model-config", str(config_path)])

    cfg = build_config(args)

    assert cfg.dataset.source.split == "train"
    assert cfg.train.train_iters == 1024
    assert cfg.train.global_batch_size == 128
    assert cfg.optimizer.lr == 3.0e-4
    assert cfg.optimizer.min_lr == 3.0e-5
    assert cfg.scheduler.lr_decay_style == "cosine"
    assert cfg.scheduler.lr_warmup_iters == 64
    assert cfg.scheduler.lr_decay_iters == cfg.train.train_iters
    assert cfg.checkpoint.save is None
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.save_interval == 0
    assert cfg.logger.tensorboard_dir is None


class _CanonicalTokenizer:
    vocab_size = 248_044
    eos_token_id = 248_046

    def __init__(self) -> None:
        self.special_tokens = {
            "<|endoftext|>": 248_044,
            "<|im_end|>": 248_046,
            "<|vision_start|>": 248_053,
            "<|vision_end|>": 248_054,
            "<|image_pad|>": 248_056,
            "<|video_pad|>": 248_057,
        }
        self.tokenizer_size = 248_077

    def __len__(self) -> int:
        return self.tokenizer_size

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_tokens[token]


def _canonical_processor() -> SimpleNamespace:
    return SimpleNamespace(tokenizer=_CanonicalTokenizer())


def test_target_multimodal_processor_contract_accepts_canonical_ids() -> None:
    _validate_processor_contract(_canonical_processor(), model_vocab_size=248_320)


def test_target_multimodal_processor_contract_rejects_tokenizer_drift() -> None:
    processor = _canonical_processor()
    processor.tokenizer.special_tokens["<|image_pad|>"] = 248_055

    with pytest.raises(ValueError, match=r"<\|image_pad\|>=248055"):
        _validate_processor_contract(processor, model_vocab_size=248_320)


def test_target_multimodal_processor_contract_allows_chat_eos() -> None:
    processor = _canonical_processor()

    _validate_processor_contract(processor, model_vocab_size=248_320)

    processor.tokenizer.eos_token_id = 248_044
    with pytest.raises(ValueError, match="eos_token_id=248044"):
        _validate_processor_contract(processor, model_vocab_size=248_320)


def test_target_multimodal_processor_contract_rejects_embedding_overflow() -> None:
    processor = _canonical_processor()
    processor.tokenizer.tokenizer_size = 248_321

    with pytest.raises(ValueError, match="tokenizer_size=248321"):
        _validate_processor_contract(processor, model_vocab_size=248_320)


def test_target_recipe_reads_nested_config_and_sizes_allocation_guards(tmp_path) -> None:
    text = _tiny_text_config()
    text["num_experts"] = 32
    text["make_ngram_vocab_size_divisible_by"] = 32
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": text}),
        encoding="utf-8",
    )

    cfg = qwenair_text_pretrain_32gpu_b300_bf16_config(path, seq_length=64)

    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.train.global_batch_size == 32
    assert cfg.model.qwenair_text_config["max_single_rank_parameters"] > 0
    assert cfg.model.qwenair_text_config["max_single_rank_ple_elements"] > 0
    assert cfg.tokenizer.vocab_size == text["vocab_size"]


def test_target_recipe_has_an_offline_default() -> None:
    cfg = qwenair_text_pretrain_32gpu_b300_bf16_config()

    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.train.global_batch_size == 32
    assert cfg.model.num_layers == 48
    assert cfg.model.num_moe_experts == 512
    assert cfg.model.hidden_size == 2560
    assert cfg.model.seq_length == 64
    assert cfg.model.qwenair_text_config["model_type"] == "qwen4_exp_text"
    assert cfg.model.qwenair_text_config["dtype"] == "bfloat16"
    assert cfg.dataset.seq_length == 64
    assert cfg.ddp.bucket_size == 40_000_000
    assert cfg.logger.log_interval == 1
    planning_config = QwenAirTextConfig.from_hf_dict(cfg.model.qwenair_text_config)
    estimate = estimate_qwenair_training_memory(planning_config, world_size=32)
    assert cfg.model.qwenair_text_config["max_single_rank_ple_elements"] == estimate.ple_parameters_per_rank
    assert cfg.model.qwenair_text_config["max_single_rank_parameters"] == estimate.routed_expert_parameters_per_rank


def test_target_recipe_rejects_the_multimodal_wrapper_as_text_config(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8")

    with pytest.raises(ValueError, match="qwen4_exp_text"):
        qwenair_text_pretrain_32gpu_b300_bf16_config(path, seq_length=64)


def test_target_recipe_rejects_eos_outside_vocabulary(tmp_path) -> None:
    text = _tiny_text_config()
    text["eos_token_id"] = text["vocab_size"]
    text["num_experts"] = 32
    text["make_ngram_vocab_size_divisible_by"] = 32
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text}), encoding="utf-8")

    with pytest.raises(ValueError, match="eos_token_id"):
        qwenair_text_pretrain_32gpu_b300_bf16_config(path, seq_length=64)
