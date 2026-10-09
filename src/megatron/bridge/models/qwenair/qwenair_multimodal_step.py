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

"""Megatron Bridge forward step for end-to-end QwenAir multimodal SFT."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.models.qwenair import QwenAirOutput
from megatron.core.utils import get_model_config

from megatron.bridge.models.qwenair.qwenair_step import qwenair_loss
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection


_QWENAIR_VISUAL_KEYS = {
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
}
_SHIFTED_LABEL_AUDIT_STATE_ATTR = "_qwenair_multimodal_shifted_label_contract_audited"


def _cuda(value: Any) -> Any:
    return value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value


def _validate_shifted_labels(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    """Validate the pre-shifted causal-LM label contract and return its effective mask."""
    if tokens.ndim != 2:
        raise ValueError(f"QwenAir multimodal tokens must have shape [batch, sequence], got {tuple(tokens.shape)}")
    if labels.shape != tokens.shape:
        raise ValueError(
            "QwenAir multimodal labels must have the same shape as tokens, "
            f"got labels={tuple(labels.shape)} and tokens={tuple(tokens.shape)}"
        )
    if loss_mask.shape != tokens.shape:
        raise ValueError(
            "QwenAir multimodal loss_mask must have the same shape as tokens, "
            f"got loss_mask={tuple(loss_mask.shape)} and tokens={tuple(tokens.shape)}"
        )
    if tokens.shape[1] == 0:
        raise ValueError("QwenAir multimodal sequences must contain at least one token")
    if vocab_size <= 0:
        raise ValueError(f"QwenAir multimodal vocab_size must be positive, got {vocab_size}")

    invalid_token = (tokens < 0) | (tokens >= vocab_size)
    if bool(invalid_token.any()):
        token_id = int(tokens[invalid_token][0].item())
        raise ValueError(f"QwenAir multimodal token {token_id} is outside [0, {vocab_size})")

    has_label = labels != -100
    out_of_range = has_label & ((labels < 0) | (labels >= vocab_size))
    if bool(out_of_range.any()):
        invalid_label = int(labels[out_of_range][0].item())
        raise ValueError(
            f"QwenAir multimodal label {invalid_label} is outside [0, {vocab_size}); only -100 may be ignored"
        )

    supervised = loss_mask.bool()
    if bool(supervised[:, -1].any()):
        raise ValueError("QwenAir multimodal shifted labels cannot supervise the final sequence position")

    valid = supervised & has_label
    misaligned = valid[:, :-1] & (labels[:, :-1] != tokens[:, 1:])
    if bool(misaligned.any()):
        raise ValueError(
            "QwenAir multimodal labels must already be shifted: every supervised label at position i "
            "must equal tokens at position i + 1"
        )
    return valid


def _shifted_label_mask(
    state: GlobalState,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    """Audit one batch per training state, then use the synchronization-free mask path."""
    if not getattr(state, _SHIFTED_LABEL_AUDIT_STATE_ATTR, False):
        valid = _validate_shifted_labels(tokens, labels, loss_mask, vocab_size=vocab_size)
        setattr(state, _SHIFTED_LABEL_AUDIT_STATE_ATTR, True)
        return valid
    return loss_mask.bool() & (labels != -100)


def _multimodal_batch(data_iterator: Iterable) -> tuple[torch.Tensor, ...]:
    batch = next(data_iterator)
    if batch.get("cu_seqlens_q") is not None or batch.get("cu_seqlens") is not None:
        raise NotImplementedError("QwenAir multimodal training does not support packed sequences")

    tokens = batch.get("tokens")
    if tokens is None:
        tokens = batch.get("input_ids")
    tokens = _cuda(tokens)
    labels = _cuda(batch.get("labels"))
    loss_mask = _cuda(batch.get("loss_mask"))
    attention_mask = _cuda(batch.get("attention_mask"))
    mm_token_type_ids = _cuda(batch.get("mm_token_type_ids"))

    visual_inputs = batch.get("visual_inputs")
    visual_kwargs: dict[str, torch.Tensor] = {}
    if visual_inputs is not None:
        for name, value in vars(visual_inputs).items():
            if value is not None:
                setattr(visual_inputs, name, _cuda(value))
        visual_kwargs = visual_inputs.normalized_for_model()
        visual_token_type_ids = visual_kwargs.pop("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            mm_token_type_ids = visual_token_type_ids
        elif visual_token_type_ids is not None and not torch.equal(mm_token_type_ids, visual_token_type_ids):
            raise ValueError("Conflicting top-level and visual-input mm_token_type_ids")
        # QwenAir derives video time positions from its timestamp tokens and
        # deliberately does not consume Qwen3-VL's second_per_grid_ts field.
        visual_kwargs.pop("second_per_grid_ts", None)
        unsupported = sorted(set(visual_kwargs).difference(_QWENAIR_VISUAL_KEYS))
        if unsupported:
            raise ValueError(f"Unsupported QwenAir visual inputs: {', '.join(unsupported)}")

    if tokens is None or labels is None or loss_mask is None:
        raise RuntimeError("QwenAir multimodal training requires input_ids, labels, and loss_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(tokens, dtype=torch.bool)
    if visual_kwargs.get("pixel_values") is None or visual_kwargs.get("image_grid_thw") is None:
        raise RuntimeError("QwenAir multimodal training requires real image pixels and image_grid_thw")
    return tokens, labels, loss_mask, attention_mask, mm_token_type_ids, visual_kwargs


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: torch.nn.Module,
    return_schedule_plan: bool = False,
) -> tuple[torch.Tensor, partial]:
    """Run one QwenAir image-text batch with EP-correct token weighting."""
    if return_schedule_plan:
        raise NotImplementedError("QwenAir multimodal training does not support overlapped schedules")
    config = get_model_config(model)
    pg_collection = get_pg_collection(model)
    tokens, labels, loss_mask, attention_mask, mm_token_type_ids, visual_kwargs = _multimodal_batch(data_iterator)

    valid = _shifted_label_mask(state, tokens, labels, loss_mask, vocab_size=int(config.vocab_size))
    valid_tokens = valid.sum(dtype=torch.int64)
    if int(valid_tokens) == 0:
        raise ValueError("QwenAir multimodal batch has no supervised assistant tokens")
    masked_labels = torch.where(valid, labels, -100)

    output = model(
        input_ids=tokens,
        attention_mask=attention_mask,
        position_ids=None,
        labels=masked_labels,
        labels_are_shifted=True,
        output_router_logits=True,
        mm_token_type_ids=mm_token_type_ids,
        **visual_kwargs,
    )
    if not isinstance(output, QwenAirOutput) or output.loss is None:
        raise RuntimeError("QwenAir multimodal model must return QwenAirOutput with a loss")

    ep_valid_tokens = valid_tokens.detach().to(dtype=torch.float32)
    dist.all_reduce(ep_valid_tokens, group=pg_collection.ep)
    ce_loss = output.loss.detach()
    if output.aux_loss is not None:
        ce_loss = ce_loss - config.router_aux_loss_coef * output.aux_loss.detach()
    loss_sum = output.loss * ep_valid_tokens
    reporting_loss_sum = ce_loss * ep_valid_tokens
    router_aux_loss_sum = None
    if output.aux_loss is not None:
        router_aux_loss_sum = output.aux_loss.detach() * valid_tokens
    return loss_sum, partial(
        qwenair_loss,
        num_tokens=valid_tokens.to(dtype=torch.int),
        reporting_loss_sum=reporting_loss_sum,
        router_aux_loss_sum=router_aux_loss_sum,
    )


__all__ = ["forward_step"]
