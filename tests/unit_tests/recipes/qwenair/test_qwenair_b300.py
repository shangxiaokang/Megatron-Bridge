# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QwenAir B300 recipe contracts."""

import json

import pytest
import torch

from megatron.bridge.models.qwenair import QwenAirModelProvider
from megatron.bridge.recipes.qwenair.b300.qwenair import (
    _tiny_text_config,
    qwenair_text_pretrain_32gpu_b300_bf16_config,
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
    assert cfg.model.qsa_backend == "te_indexed_sdpa"
    assert cfg.model.mtp_num_layers == 0
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.train.global_batch_size == 8
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seq_length == 64
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.mixed_precision.bf16 is True
    assert cfg.optimizer.main_params_dtype == torch.float32


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


def test_target_recipe_rejects_the_multimodal_wrapper_as_text_config(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8")

    with pytest.raises(ValueError, match="qwen4_exp_text"):
        qwenair_text_pretrain_32gpu_b300_bf16_config(path, seq_length=64)
