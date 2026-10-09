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

import pytest
import torch

import megatron.bridge.models.qwenair.qwenair_multimodal_step as qwenair_multimodal_step
from megatron.bridge.models.qwenair.qwenair_multimodal_step import _multimodal_batch, _validate_shifted_labels
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


pytestmark = pytest.mark.unit


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


def test_validate_shifted_labels_returns_only_effectively_supervised_positions() -> None:
    tokens = torch.tensor([[7, 11, 12, 13, 14]])
    labels = torch.tensor([[11, 99, 13, -100, -100]])
    loss_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]])

    valid = _validate_shifted_labels(tokens, labels, loss_mask, vocab_size=100)

    assert valid.tolist() == [[True, False, True, False, False]]


@pytest.mark.parametrize(
    ("tokens", "labels", "loss_mask", "error"),
    [
        (
            torch.tensor([7, 8, 9]),
            torch.tensor([8, 9, -100]),
            torch.tensor([1.0, 1.0, 0.0]),
            "tokens must have shape",
        ),
        (
            torch.tensor([[7, 8, 9]]),
            torch.tensor([[8, -100]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            "labels must have the same shape",
        ),
        (
            torch.tensor([[7, 8, 9]]),
            torch.tensor([[8, 9, -100]]),
            torch.tensor([[1.0, 0.0]]),
            "loss_mask must have the same shape",
        ),
    ],
)
def test_validate_shifted_labels_rejects_invalid_shapes(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _validate_shifted_labels(tokens, labels, loss_mask, vocab_size=100)


@pytest.mark.parametrize("invalid_label", [-1, 100])
def test_validate_shifted_labels_rejects_nonignored_labels_outside_vocabulary(invalid_label: int) -> None:
    tokens = torch.tensor([[7, 8, 9]])
    labels = torch.tensor([[8, invalid_label, -100]])
    loss_mask = torch.tensor([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match=rf"label {invalid_label} is outside \[0, 100\)"):
        _validate_shifted_labels(tokens, labels, loss_mask, vocab_size=100)


@pytest.mark.parametrize("invalid_token", [-1, 100])
def test_validate_shifted_labels_rejects_tokens_outside_vocabulary(invalid_token: int) -> None:
    tokens = torch.tensor([[invalid_token, 8, 9]])
    labels = torch.tensor([[8, 9, -100]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])

    with pytest.raises(ValueError, match=rf"token {invalid_token} is outside \[0, 100\)"):
        _validate_shifted_labels(tokens, labels, loss_mask, vocab_size=100)


def test_validate_shifted_labels_rejects_unshifted_supervised_label() -> None:
    tokens = torch.tensor([[7, 8, 9]])
    labels = torch.tensor([[7, 9, -100]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])

    with pytest.raises(ValueError, match="labels must already be shifted"):
        _validate_shifted_labels(tokens, labels, loss_mask, vocab_size=100)


def test_validate_shifted_labels_rejects_supervision_at_final_position() -> None:
    tokens = torch.tensor([[7, 8, 9]])
    labels = torch.tensor([[8, 9, -100]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="cannot supervise the final sequence position"):
        _validate_shifted_labels(tokens, labels, loss_mask, vocab_size=100)


def test_shifted_label_mask_runs_full_contract_audit_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    state = type("State", (), {})()
    tokens = torch.tensor([[7, 8, 9]])
    labels = torch.tensor([[8, 9, -100]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    audit_calls = 0
    validator = qwenair_multimodal_step._validate_shifted_labels

    def counted_validator(*args, **kwargs):
        nonlocal audit_calls
        audit_calls += 1
        return validator(*args, **kwargs)

    monkeypatch.setattr(qwenair_multimodal_step, "_validate_shifted_labels", counted_validator)

    first = qwenair_multimodal_step._shifted_label_mask(state, tokens, labels, loss_mask, vocab_size=100)
    second = qwenair_multimodal_step._shifted_label_mask(state, tokens, labels, loss_mask, vocab_size=100)

    assert audit_calls == 1
    assert torch.equal(first, second)
