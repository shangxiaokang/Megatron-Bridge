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

"""Single-rank image/video input composition and gradient smoke tests."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.qwenair import QwenAirForCausalLM, QwenAirTextConfig
from torch import nn

from megatron.bridge.models.qwenair.multimodal_model import QwenAirForConditionalGeneration
from megatron.bridge.models.qwenair.vision_reference import QwenAirVisionTextReference


class _TinyVisual(nn.Module):
    """Trainable feature producer with the oracle's patches-per-token shape."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(24, 16)

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor) -> SimpleNamespace:
        assert pixel_values.shape[0] == int(grid_thw.prod())
        features = self.proj(pixel_values).reshape(-1, 4, 16).mean(dim=1)
        return SimpleNamespace(pooler_output=features)


class _TinyConditionalVisual(nn.Module):
    """Production-wrapper-shaped vision stub with an optional graph break."""

    def __init__(self, *, detach_output: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(24, 16)
        self.detach_output = detach_output

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor) -> torch.Tensor:
        assert pixel_values.shape[0] == int(grid_thw.prod())
        features = self.proj(pixel_values).reshape(-1, 4, 16).mean(dim=1)
        return features.detach() if self.detach_output else features


def _tiny_text_model() -> QwenAirForCausalLM:
    config = QwenAirTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=3,
        hc_count=4,
        hc_lowrank=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        indexer_n_heads=2,
        indexer_head_dim=4,
        indexer_budget=4,
        indexer_compress_ratio=2,
        partial_rotary_factor=0.5,
        mrope_section=(1, 1, 0),
        ple_layer_ids=[1],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=8,
        eos_token_id=3,
        mtp_num_hidden_layers=0,
    )
    return QwenAirForCausalLM(config)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_visual_feature_scatter_trains_vision_and_text(modality: str, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(1920)
    # MCore's strict FP32 QSA reference requires IEEE matmuls on CUDA.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual = _TinyVisual()
    model = QwenAirVisionTextReference(
        _tiny_text_model(), visual, image_token_id=31, video_token_id=32, spatial_merge_size=2
    ).to(device)
    if modality == "image":
        ids = torch.tensor([[7, 31, 31, 8, 9]], device=device)
        token_types = torch.tensor([[0, 1, 1, 0, 0]], device=device)
        pixels = torch.linspace(-0.2, 0.3, steps=8 * 24, device=device).reshape(8, 24).requires_grad_()
        kwargs = {"pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 2, 4]], device=device)}
    else:
        ids = torch.tensor([[7, 32, 32, 11, 32, 32, 9]], device=device)
        token_types = torch.tensor([[0, 2, 2, 0, 2, 2, 0]], device=device)
        pixels = torch.linspace(-0.3, 0.4, steps=16 * 24, device=device).reshape(16, 24).requires_grad_()
        kwargs = {"pixel_values_videos": pixels, "video_grid_thw": torch.tensor([[2, 2, 4]], device=device)}
    labels = ids.clone()
    labels[token_types != 0] = -100
    output = model(ids, labels=labels, mm_token_type_ids=token_types, **kwargs)
    assert output.logits.shape == (*ids.shape, 64)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert pixels.grad is not None and pixels.grad.abs().sum() > 0
    assert visual.proj.weight.grad is not None and visual.proj.weight.grad.abs().sum() > 0
    ple_weight = model.language_model.model.layers[0].ple.ple_embedding.ngram_embedding.weight
    assert ple_weight.grad is not None and ple_weight.grad.abs().sum() > 0
    assert model.language_model.model.embed_tokens.weight.grad is not None


def test_multimodal_positions_require_modality_ids() -> None:
    model = QwenAirVisionTextReference(
        _tiny_text_model(), _TinyVisual(), image_token_id=31, video_token_id=32, spatial_merge_size=2
    )
    with pytest.raises(ValueError, match="mm_token_type_ids"):
        model(
            torch.tensor([[7, 31, 31, 8, 9]]),
            pixel_values=torch.zeros(8, 24),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
        )


@pytest.mark.parametrize("detach_output", [False, True])
def test_visual_gradient_audit_rejects_a_detached_vision_branch(
    detach_output: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first audited backward must run even when vision has no loss path."""
    torch.manual_seed(923)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    visual = _TinyConditionalVisual(detach_output=detach_output)
    model = QwenAirForConditionalGeneration(
        _tiny_text_model(),
        visual,
        image_token_id=31,
        video_token_id=32,
        spatial_merge_size=2,
        audit_visual_gradient=True,
    )
    input_ids = torch.tensor([[7, 31, 31, 8, 9]])
    output = model(
        input_ids,
        labels=torch.tensor([[31, 31, 8, 9, -100]]),
        pixel_values=torch.linspace(-0.2, 0.3, steps=8 * 24).reshape(8, 24),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
    )
    assert output.loss is not None

    if detach_output:
        with pytest.raises(RuntimeError, match="vision gradient is zero"):
            output.loss.backward()
        assert model._visual_gradient_audited is False
    else:
        output.loss.backward()
        assert model._visual_gradient_audited is True
        assert visual.proj.weight.grad is not None
        assert visual.proj.weight.grad.norm() > 0
