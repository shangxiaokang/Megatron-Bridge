# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QwenAir B300 recipe contracts."""

import json
from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.qwenair import QwenAirTextConfig, estimate_qwenair_training_memory

from megatron.bridge.models.qwenair import QwenAirModelProvider, QwenAirMultimodalModelProvider, qwenair_provider
from megatron.bridge.recipes.qwenair.b300.qwenair import (
    _tiny_text_config,
    configure_qwenair_indexed_data,
    qwenair_text_pretrain_32gpu_b300_bf16_config,
    qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config,
    qwenair_tiny_pretrain_8gpu_b300_bf16_config,
)


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
    assert cfg.dataset.source.load_kwargs == {"revision": "dataset-commit"}
    assert cfg.dataset.source_weights == [1.0]
    assert cfg.dataset.hf_processor_kwargs == {"revision": "processor-commit"}
    assert cfg.dataset.min_pixels == 224 * 224
    assert cfg.dataset.max_pixels == 224 * 224
    assert cfg.train.train_iters == 128
    assert cfg.train.global_batch_size == 128
    assert cfg.scheduler.lr_warmup_iters == 12


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
