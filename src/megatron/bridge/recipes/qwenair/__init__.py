# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QwenAir training recipes."""

import importlib.util


__all__ = []

# Keep the general recipe package importable with an external MCore checkout
# that does not yet provide QwenAir. This matches the model package boundary.
if importlib.util.find_spec("megatron.core.models.qwenair") is not None:
    from megatron.bridge.recipes.qwenair.b300 import (
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
