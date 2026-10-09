# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""B300 QwenAir training recipes."""

from .qwenair import (
    configure_qwenair_indexed_data,
    qwenair_target_multimodal_finetune_32gpu_b300_bf16_config,
    qwenair_text_pretrain_32gpu_b300_bf16_config,
    qwenair_tiny_pretrain_8gpu_b300_bf16_config,
)


__all__ = [
    "configure_qwenair_indexed_data",
    "qwenair_target_multimodal_finetune_32gpu_b300_bf16_config",
    "qwenair_text_pretrain_32gpu_b300_bf16_config",
    "qwenair_tiny_pretrain_8gpu_b300_bf16_config",
]
