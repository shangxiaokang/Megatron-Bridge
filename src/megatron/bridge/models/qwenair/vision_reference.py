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

"""Unregistered single-rank QwenAir vision/text reference input composition.

The vision encoder is injected so that a frozen HF model can be used as an
oracle. This module does not provide a production vision encoder or checkpoint
mapping and must not be registered as support for ``model_type=qwen4_exp``.
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn


if TYPE_CHECKING:
    from megatron.core.models.qwenair import QwenAirForCausalLM, QwenAirOutput


def scatter_qwenair_visual_features(
    input_ids: Tensor,
    inputs_embeds: Tensor,
    *,
    image_token_id: int,
    video_token_id: int,
    image_features: Tensor | None = None,
    video_features: Tensor | None = None,
) -> Tensor:
    """Replace provided modality placeholders with one feature row per token."""
    if input_ids.ndim != 2 or inputs_embeds.ndim != 3 or inputs_embeds.shape[:2] != input_ids.shape:
        raise ValueError("input_ids and inputs_embeds must have shapes [B,S] and [B,S,H]")
    hidden_size = inputs_embeds.shape[-1]
    for token_id, features, name in (
        (image_token_id, image_features, "image"),
        (video_token_id, video_features, "video"),
    ):
        if features is None:
            continue
        if features.ndim != 2 or features.shape[-1] != hidden_size:
            raise ValueError(f"{name} features must have shape [tokens,{hidden_size}]")
        placeholder_mask = input_ids == token_id
        placeholder_count = int(placeholder_mask.sum())
        if placeholder_count != features.shape[0]:
            raise ValueError(
                f"{name} features and placeholders differ: {features.shape[0]} features, "
                f"{placeholder_count} placeholders"
            )
        features = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(placeholder_mask.unsqueeze(-1), features)
    return inputs_embeds


def qwenair_multimodal_position_ids(
    input_ids: Tensor,
    mm_token_type_ids: Tensor,
    *,
    spatial_merge_size: int,
    image_grid_thw: Tensor | None = None,
    video_grid_thw: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build HF Qwen4-Exp grouped image/video positions and RoPE deltas."""
    if input_ids.ndim != 2 or mm_token_type_ids.shape != input_ids.shape:
        raise ValueError("input_ids and mm_token_type_ids must both have shape [B,S]")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have shape [B,S]")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")

    if video_grid_thw is not None:
        if video_grid_thw.ndim != 2 or video_grid_thw.shape[-1] != 3:
            raise ValueError("video_grid_thw must have shape [videos,3]")
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    if image_grid_thw is not None and (image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3):
        raise ValueError("image_grid_thw must have shape [images,3]")

    grid_iterators = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }
    positions = torch.zeros((3, *input_ids.shape), dtype=input_ids.dtype, device=input_ids.device)
    deltas = []
    for batch_index, token_types in enumerate(mm_token_type_ids):
        keep = attention_mask[batch_index].bool() if attention_mask is not None else None
        if keep is not None:
            token_types = token_types[keep]
        current_position = 0
        groups = []
        for modality, entries in groupby(token_types.tolist()):
            group_length = sum(1 for _ in entries)
            if modality == 0:
                group_positions = torch.arange(group_length, device=input_ids.device)
                groups.append(group_positions.expand(3, -1) + current_position)
                current_position += group_length
                continue
            if modality not in (1, 2) or grid_iterators[modality] is None:
                raise ValueError(f"Missing grid for modality type {modality}")
            grid = next(grid_iterators[modality], None)
            if grid is None:
                raise ValueError(f"Not enough grid entries for modality type {modality}")
            temporal, height, width = (int(value) for value in grid)
            if height % spatial_merge_size or width % spatial_merge_size:
                raise ValueError("Vision grid must be divisible by spatial_merge_size")
            merged_height = height // spatial_merge_size
            merged_width = width // spatial_merge_size
            if temporal * merged_height * merged_width != group_length:
                raise ValueError("Vision token group length does not match its grid")
            time_grid, height_grid, width_grid = torch.meshgrid(
                torch.arange(temporal, device=input_ids.device),
                torch.arange(merged_height, device=input_ids.device),
                torch.arange(merged_width, device=input_ids.device),
                indexing="ij",
            )
            group_positions = torch.stack((time_grid, height_grid, width_grid)).reshape(3, -1)
            group_positions += current_position
            groups.append(group_positions)
            current_position += max(height, width) // spatial_merge_size

        if not groups:
            raise ValueError("Each sequence must contain at least one unmasked token")
        sequence_positions = torch.cat(groups, dim=1)
        if keep is None:
            positions[:, batch_index] = sequence_positions
        else:
            positions[:, batch_index, keep] = sequence_positions
        deltas.append(sequence_positions.max() + 1 - token_types.numel())
    for modality, grids in grid_iterators.items():
        if grids is not None and next(grids, None) is not None:
            raise ValueError(f"Unused grid entries for modality type {modality}")
    return positions, torch.stack(deltas).unsqueeze(1)


class QwenAirVisionTextReference(nn.Module):
    """Combine an injected vision module with the MCore QwenAir text reference."""

    def __init__(
        self,
        language_model: QwenAirForCausalLM,
        visual: nn.Module,
        *,
        image_token_id: int,
        video_token_id: int,
        spatial_merge_size: int,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.visual = visual
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.spatial_merge_size = spatial_merge_size

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        pixel_values: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        mm_token_type_ids: Tensor | None = None,
        output_router_logits: bool = False,
    ) -> QwenAirOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,S]")
        image_features = None
        video_features = None
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("pixel_values requires image_grid_thw")
            image_features = self.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("pixel_values_videos requires video_grid_thw")
            video_features = self.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        inputs_embeds = scatter_qwenair_visual_features(
            input_ids,
            inputs_embeds,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            image_features=image_features,
            video_features=video_features,
        )
        if position_ids is None and (image_grid_thw is not None or video_grid_thw is not None):
            if mm_token_type_ids is None:
                raise ValueError("Multimodal grids require mm_token_type_ids for 3D M-RoPE")
            position_ids, _ = qwenair_multimodal_position_ids(
                input_ids,
                mm_token_type_ids,
                spatial_merge_size=self.spatial_merge_size,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        return self.language_model(
            None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_router_logits=output_router_logits,
            inputs_embeds=inputs_embeds,
            ple_input_ids=input_ids,
        )
