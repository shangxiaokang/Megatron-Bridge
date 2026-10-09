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

"""QwenAir multimodal training-step batch contracts."""

import torch

from megatron.bridge.models.qwenair.qwenair_multimodal_step import _multimodal_batch
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


class _NoCudaTensor(torch.Tensor):
    def cuda(self, non_blocking: bool = False) -> torch.Tensor:  # type: ignore[override]
        del non_blocking
        return self


def _as_nocuda(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.as_subclass(_NoCudaTensor)


def test_multimodal_batch_derives_token_types_in_model_when_processor_omits_them() -> None:
    """Qwen3VLProcessor does not need to emit optional ``mm_token_type_ids``."""
    tokens = _as_nocuda(torch.tensor([[7, 31, 31, 8]], dtype=torch.long))
    pixels = _as_nocuda(torch.randn(4, 24))
    image_grid_thw = _as_nocuda(torch.tensor([[1, 2, 2]], dtype=torch.long))
    second_per_grid_ts = _as_nocuda(torch.tensor([0.5]))
    batch = {
        "input_ids": tokens,
        "labels": _as_nocuda(torch.tensor([[31, 31, 8, -100]], dtype=torch.long)),
        "loss_mask": _as_nocuda(torch.tensor([[0.0, 0.0, 1.0, 0.0]])),
        "attention_mask": _as_nocuda(torch.ones_like(tokens, dtype=torch.bool)),
        "visual_inputs": GenericVisualInputs(
            pixel_values=pixels,
            image_grid_thw=image_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        ),
    }

    result = _multimodal_batch(iter([batch]))

    result_tokens, _, _, _, mm_token_type_ids, visual_kwargs = result
    assert result_tokens is tokens
    assert mm_token_type_ids is None
    assert visual_kwargs["pixel_values"] is pixels
    assert visual_kwargs["image_grid_thw"] is image_grid_thw
    assert "second_per_grid_ts" not in visual_kwargs
