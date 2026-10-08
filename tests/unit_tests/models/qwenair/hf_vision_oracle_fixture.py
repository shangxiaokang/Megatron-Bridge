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

"""Generate a tiny image/video feature-scatter fixture with the frozen HF oracle.

Run in a separate process with ``QWENAIR_HF_ORACLE_SRC`` prepended to PYTHONPATH.
The fixture deliberately stops before the QwenAir language model; it captures
the vision and multimodal input contract without implying VLM training support.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


_ORACLE_FILES = {
    "modeling_qwen4_exp.py": "f17e23c244294ef990b7636560394058c691cecbf6b9e273874aabcae43a6cbe",
    "configuration_qwen4_exp.py": "26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254",
}


def _checked_oracle_source() -> Path:
    source = Path(os.environ["QWENAIR_HF_ORACLE_SRC"]).resolve()
    model_dir = source / "transformers" / "models" / "qwen4_exp"
    for name, expected in _ORACLE_FILES.items():
        digest = hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Frozen Qwen4-Exp oracle source mismatch: {name}")
    return source


def _position_stub(vision_config, model_class):
    """Call the frozen wrapper's pure M-RoPE methods without building its LLM."""
    stub = SimpleNamespace(config=SimpleNamespace(vision_config=vision_config))
    stub.get_vision_position_ids = MethodType(model_class.get_vision_position_ids, stub)
    return stub


def _case(*, vision, model_class, modality: str) -> dict[str, torch.Tensor]:
    image_token_id, video_token_id = 31, 32
    if modality == "image":
        input_ids = torch.tensor([[7, image_token_id, image_token_id, 8, 9]])
        token_types = torch.tensor([[0, 1, 1, 0, 0]])
        grid = torch.tensor([[1, 2, 4]])
        pixels = torch.linspace(-0.2, 0.3, steps=8 * 24).reshape(8, 24).requires_grad_()
        feature_arg = "image_features"
        grid_arg = "image_grid_thw"
        placeholder = image_token_id
    else:
        # Two frames are separated by a text/timestamp token, as in the HF M-RoPE contract.
        input_ids = torch.tensor([[7, video_token_id, video_token_id, 11, video_token_id, video_token_id, 9]])
        token_types = torch.tensor([[0, 2, 2, 0, 2, 2, 0]])
        grid = torch.tensor([[2, 2, 4]])
        pixels = torch.linspace(-0.3, 0.4, steps=16 * 24).reshape(16, 24).requires_grad_()
        feature_arg = "video_features"
        grid_arg = "video_grid_thw"
        placeholder = video_token_id

    pooled = vision(pixels, grid_thw=grid).pooler_output
    expected_feature_count = int(grid.prod() // vision.spatial_merge_size**2)
    assert pooled.shape == (expected_feature_count, vision.config.out_hidden_size)

    embedding = torch.randn(1, input_ids.shape[1], vision.config.out_hidden_size, requires_grad=True)
    stub = _position_stub(vision.config, model_class)
    stub.config.image_token_id = image_token_id
    stub.config.video_token_id = video_token_id
    masks = model_class.get_placeholder_mask(stub, input_ids, embedding, **{feature_arg: pooled})
    mask = masks[0] if modality == "image" else masks[1]
    scattered = embedding.masked_scatter(mask, pooled)
    assert torch.equal(scattered[0, input_ids[0] == placeholder], pooled)
    assert torch.equal(scattered[0, input_ids[0] != placeholder], embedding[0, input_ids[0] != placeholder])

    positions, rope_delta = model_class.get_rope_index(stub, input_ids, token_types, **{grid_arg: grid})
    assert positions.shape == (3, 1, input_ids.shape[1])
    assert rope_delta.shape == (1, 1)

    loss = scattered.square().sum()
    loss.backward()
    assert pixels.grad is not None and pixels.grad.abs().sum() > 0
    assert vision.patch_embed.proj.weight.grad is not None
    assert vision.patch_embed.proj.weight.grad.abs().sum() > 0
    assert embedding.grad is not None
    assert torch.count_nonzero(embedding.grad[0, input_ids[0] == placeholder]) == 0

    return {
        "input_ids": input_ids,
        "mm_token_type_ids": token_types,
        "grid_thw": grid,
        "pixel_values": pixels.detach(),
        "pixel_grad": pixels.grad.detach(),
        "input_embeddings": embedding.detach(),
        "features": pooled.detach(),
        "scattered_embeddings": scattered.detach(),
        "position_ids": positions.detach(),
        "rope_delta": rope_delta.detach(),
        "loss": loss.detach(),
    }


def main() -> None:
    source = _checked_oracle_source()
    import transformers
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel, Qwen4ExpVisionModel

    if not Path(transformers.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Frozen HF oracle source is not first on PYTHONPATH")

    torch.manual_seed(1916)
    torch.set_num_threads(1)
    config = Qwen4ExpVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    vision = Qwen4ExpVisionModel(config).eval()
    state = {name: value.detach().clone() for name, value in vision.state_dict().items()}
    image = _case(vision=vision, model_class=Qwen4ExpModel, modality="image")
    vision.zero_grad(set_to_none=True)
    video = _case(vision=vision, model_class=Qwen4ExpModel, modality="video")

    output_path = os.environ.get("QWENAIR_HF_ORACLE_OUTPUT")
    if output_path:
        torch.save({"vision_state": state, "image": image, "video": video}, output_path)
    print(
        json.dumps(
            {
                "oracle": "2ff8a4b2752cb54ff8dedfd7408ac7e6b7d2ee40",
                "image_features": list(image["features"].shape),
                "video_features": list(video["features"].shape),
                "image_position_ids": image["position_ids"].tolist(),
                "video_position_ids": video["position_ids"].tolist(),
                "image_pixel_grad_l1": float(image["pixel_grad"].abs().sum()),
                "video_pixel_grad_l1": float(video["pixel_grad"].abs().sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
