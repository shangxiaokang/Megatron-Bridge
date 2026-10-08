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

"""Optional frozen-HF fixture for QwenAir image/video input semantics."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from megatron.bridge.models.qwenair.vision_reference import (
    qwenair_multimodal_position_ids,
    scatter_qwenair_visual_features,
)


def test_frozen_hf_vision_and_feature_scatter_oracle(tmp_path: Path) -> None:
    source = os.environ.get("QWENAIR_HF_ORACLE_SRC")
    if not source:
        pytest.skip("Set QWENAIR_HF_ORACLE_SRC to the frozen 2ff8a4b275 Transformers src directory")

    fixture = Path(__file__).with_name("hf_vision_oracle_fixture.py")
    output = tmp_path / "qwenair_vision_oracle.pt"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
    env["QWENAIR_HF_ORACLE_OUTPUT"] = str(output)
    oracle_python = os.environ.get("QWENAIR_HF_ORACLE_PYTHON", sys.executable)
    result = subprocess.run(
        [oracle_python, str(fixture)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["oracle"] == "2ff8a4b2752cb54ff8dedfd7408ac7e6b7d2ee40"
    assert report["image_features"] == [2, 16]
    assert report["video_features"] == [4, 16]
    assert report["image_position_ids"] == [
        [[0, 1, 1, 3, 4]],
        [[0, 1, 1, 3, 4]],
        [[0, 1, 2, 3, 4]],
    ]
    assert report["video_position_ids"] == [
        [[0, 1, 1, 3, 4, 4, 6]],
        [[0, 1, 1, 3, 4, 4, 6]],
        [[0, 1, 2, 3, 4, 5, 6]],
    ]
    assert report["image_pixel_grad_l1"] > 0
    assert report["video_pixel_grad_l1"] > 0

    bundle = torch.load(output, weights_only=True)
    assert any(name.endswith("patch_embed.proj.weight") for name in bundle["vision_state"])
    for modality, token_id in (("image", 31), ("video", 32)):
        case = bundle[modality]
        token_mask = case["input_ids"][0] == token_id
        torch.testing.assert_close(case["scattered_embeddings"][0, token_mask], case["features"])
        torch.testing.assert_close(
            case["scattered_embeddings"][0, ~token_mask], case["input_embeddings"][0, ~token_mask]
        )
        assert case["pixel_grad"].abs().sum() > 0

        images = case["features"] if modality == "image" else None
        videos = case["features"] if modality == "video" else None
        scattered = scatter_qwenair_visual_features(
            case["input_ids"],
            case["input_embeddings"],
            image_token_id=31,
            video_token_id=32,
            image_features=images,
            video_features=videos,
        )
        torch.testing.assert_close(scattered, case["scattered_embeddings"], rtol=0, atol=0)
        # Frozen HF only validates and replaces a modality when its feature tensor is supplied.
        untouched = scatter_qwenair_visual_features(
            case["input_ids"], case["input_embeddings"], image_token_id=31, video_token_id=32
        )
        torch.testing.assert_close(untouched, case["input_embeddings"], rtol=0, atol=0)
        grid_arg = {"image_grid_thw": case["grid_thw"]} if images is not None else {"video_grid_thw": case["grid_thw"]}
        positions, delta = qwenair_multimodal_position_ids(
            case["input_ids"],
            case["mm_token_type_ids"],
            spatial_merge_size=2,
            **grid_arg,
        )
        torch.testing.assert_close(positions, case["position_ids"], rtol=0, atol=0)
        torch.testing.assert_close(delta, case["rope_delta"], rtol=0, atol=0)

        with pytest.raises(ValueError, match="features and placeholders differ"):
            scatter_qwenair_visual_features(
                case["input_ids"],
                case["input_embeddings"],
                image_token_id=31,
                video_token_id=32,
                image_features=case["features"][:1] if modality == "image" else None,
                video_features=case["features"][:1] if modality == "video" else None,
            )
