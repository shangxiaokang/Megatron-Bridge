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

"""Native QwenAir vision-to-language composition for training.

QwenAir and Qwen3.5-VL use the same ViT patch/merger contract.  This module
reuses Bridge's MCore/Transformer Engine Qwen3 vision encoder, scatters its
merged features into QwenAir placeholders, preserves the original token IDs
for PLE, and constructs QwenAir's three-axis M-RoPE positions.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from megatron.core.models.qwenair import QwenAirForCausalLM, QwenAirOutput
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.qwenair.vision_reference import (
    qwenair_multimodal_position_ids,
    scatter_qwenair_visual_features,
)
from megatron.bridge.utils.common_utils import print_rank_0


class QwenAirForConditionalGeneration(MegatronModule):
    """End-to-end QwenAir image/video conditional language model.

    Pipeline, tensor, and context parallel splits are deliberately rejected by
    the provider. Expert parallelism remains owned by ``language_model`` while
    the dense vision encoder is replicated with the other dense parameters.
    """

    def __init__(
        self,
        language_model: QwenAirForCausalLM,
        vision_model: MegatronModule,
        *,
        image_token_id: int,
        video_token_id: int,
        spatial_merge_size: int,
        audit_visual_gradient: bool = False,
    ) -> None:
        super().__init__(config=language_model.config)
        self.language_model = language_model
        self.vision_model = vision_model
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.spatial_merge_size = int(spatial_merge_size)
        self.pg_collection = language_model.pg_collection
        self.tp_group = self.pg_collection.tp if self.pg_collection is not None else None
        self.share_embeddings_and_output_weights = language_model.config.tie_word_embeddings
        self._audit_visual_gradient = bool(audit_visual_gradient)
        self._visual_gradient_audited = False
        self._visual_batch_audited = False
        self._visual_gradient_sentinel_name: str | None = None
        self._visual_gradient_hook_handle: Any | None = None
        if self._audit_visual_gradient:
            sentinel = next(
                ((name, value) for name, value in vision_model.named_parameters() if value.requires_grad),
                None,
            )
            if sentinel is None:
                raise ValueError("Visual-gradient audit requires a trainable vision parameter")
            self._visual_gradient_sentinel_name = sentinel[0]

    def _audit_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        if self._visual_gradient_audited:
            return gradient
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("QwenAir vision gradient contains NaN or Inf")
        gradient_norm = gradient.float().norm()
        if float(gradient_norm) == 0.0:
            raise RuntimeError("QwenAir vision gradient is zero on the first backward pass")
        self._visual_gradient_audited = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            print_rank_0(f"QWENAIR_MM_AUDIT vision_gradient_norm={float(gradient_norm):.8e}")
        return gradient

    def _attach_visual_gradient_audit(self, output: QwenAirOutput) -> QwenAirOutput:
        """Make a missing vision gradient fail the first audited backward pass.

        A parameter hook alone is insufficient because PyTorch never calls it
        when the vision branch is detached from the language loss.  A zero
        coefficient dependency makes the sentinel participate in autograd
        without changing the loss.  The hook then observes the real accumulated
        gradient when the branch is connected, or an all-zero gradient when it
        is disconnected.  Register lazily so the hook targets the parameter
        after Bridge has moved and cast the assembled model.
        """
        if not self._audit_visual_gradient or self._visual_gradient_audited or output.loss is None:
            return output
        if self._visual_gradient_sentinel_name is None:
            raise RuntimeError("Visual-gradient audit has no sentinel parameter")
        sentinel = self.vision_model.get_parameter(self._visual_gradient_sentinel_name)
        if self._visual_gradient_hook_handle is None:
            self._visual_gradient_hook_handle = sentinel.register_hook(self._audit_gradient)
        output.loss = output.loss + sentinel.reshape(-1)[0] * 0.0
        return output

    def shared_embedding_or_output_weight(self) -> torch.Tensor | None:
        """Expose the decoder's tied embedding for MCore gradient finalization."""
        if not self.share_embeddings_and_output_weights:
            return None
        return self.language_model.model.embed_tokens.weight

    def set_input_tensor(self, input_tensor: Any) -> None:
        """Delegate the PP=1 empty-input contract to the QwenAir decoder."""
        self.language_model.set_input_tensor(input_tensor)

    @staticmethod
    def _token_types(input_ids: torch.Tensor, image_token_id: int, video_token_id: int) -> torch.Tensor:
        token_types = torch.zeros_like(input_ids, dtype=torch.int32)
        token_types.masked_fill_(input_ids == image_token_id, 1)
        token_types.masked_fill_(input_ids == video_token_id, 2)
        return token_types

    def _encode_visual(
        self,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        *,
        name: str,
    ) -> torch.Tensor | None:
        if pixel_values is None and grid_thw is None:
            return None
        if pixel_values is None or grid_thw is None:
            raise ValueError(f"{name} pixel values and grid_thw must be provided together")
        if pixel_values.ndim != 2:
            raise ValueError(f"{name} pixel values must have shape [patches, patch_width]")
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError(f"{name} grid_thw must have shape [items, 3]")
        encoded = self.vision_model(pixel_values, grid_thw=grid_thw)
        features = encoded[0] if isinstance(encoded, tuple) else encoded
        if features.ndim != 2 or features.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"{name} encoder must return [tokens,{self.config.hidden_size}], got {tuple(features.shape)}"
            )
        return features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        output_router_logits: bool = False,
        labels_are_shifted: bool = True,
    ) -> QwenAirOutput:
        """Encode real media and run the QwenAir causal language objective."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have shape [batch, sequence]")

        image_features = self._encode_visual(pixel_values, image_grid_thw, name="image")
        video_features = self._encode_visual(pixel_values_videos, video_grid_thw, name="video")
        has_visual = image_features is not None or video_features is not None
        if not has_visual:
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                labels_are_shifted=labels_are_shifted,
                output_router_logits=output_router_logits,
            )

        derived_types = self._token_types(input_ids, self.image_token_id, self.video_token_id)
        if mm_token_type_ids is None:
            mm_token_type_ids = derived_types
        else:
            if mm_token_type_ids.shape != input_ids.shape:
                raise ValueError("mm_token_type_ids must have shape [batch, sequence]")
            mm_token_type_ids = mm_token_type_ids.to(device=input_ids.device, dtype=torch.int32)
            if not torch.equal(mm_token_type_ids, derived_types):
                raise ValueError("mm_token_type_ids disagree with QwenAir image/video placeholders")

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        inputs_embeds = scatter_qwenair_visual_features(
            input_ids,
            inputs_embeds,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            image_features=image_features,
            video_features=video_features,
        )
        if position_ids is None or position_ids.ndim != 3:
            position_ids, _ = qwenair_multimodal_position_ids(
                input_ids,
                mm_token_type_ids,
                spatial_merge_size=self.spatial_merge_size,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

        if not self._visual_batch_audited:
            image_placeholders = int((input_ids == self.image_token_id).sum())
            image_feature_count = 0 if image_features is None else image_features.shape[0]
            video_placeholders = int((input_ids == self.video_token_id).sum())
            video_feature_count = 0 if video_features is None else video_features.shape[0]
            valid_tokens = input_ids.numel() if attention_mask is None else int(attention_mask.sum())
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank_0(
                    "QWENAIR_MM_AUDIT "
                    f"batch={input_ids.shape[0]} sequence={input_ids.shape[1]} valid_tokens={valid_tokens} "
                    f"image_placeholders={image_placeholders} image_features={image_feature_count} "
                    f"video_placeholders={video_placeholders} video_features={video_feature_count} "
                    f"pixel_abs_mean={float(pixel_values.float().abs().mean()) if pixel_values is not None else 0.0:.8f}"
                )
            self._visual_batch_audited = True

        # A fully valid mask is redundant. MCore validates any remaining mask
        # as a contiguous right-padded suffix before entering TE QSA.
        text_attention_mask = attention_mask
        if attention_mask is not None and bool(torch.all(attention_mask)):
            text_attention_mask = None
        output = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            ple_input_ids=input_ids,
            attention_mask=text_attention_mask,
            position_ids=position_ids,
            labels=labels,
            labels_are_shifted=labels_are_shifted,
            output_router_logits=output_router_logits,
        )
        return self._attach_visual_gradient_audit(output)


__all__ = ["QwenAirForConditionalGeneration"]
