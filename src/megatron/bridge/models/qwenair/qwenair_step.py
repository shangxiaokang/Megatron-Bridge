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

"""Megatron Bridge forward step for QwenAir EP training."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import torch
import torch.distributed as dist
from megatron.core.models.qwenair import QwenAirOutput
from megatron.core.utils import get_model_config

from megatron.bridge.training.gpt_step import get_batch
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection


def qwenair_loss(
    loss_sum: torch.Tensor,
    *,
    num_tokens: torch.Tensor,
    reporting_loss_sum: torch.Tensor,
    router_aux_loss_sum: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return an unnormalized loss sum for MCore's per-token loss path.

    ``finalize_model_grads`` sums ``num_tokens`` across DP and applies the one
    global normalization after dense WORLD and sharded EDP gradient reduction.
    This remains correct when EP replicas or accumulated microbatches contain
    different numbers of valid labels.
    """
    metrics = {"lm loss": torch.stack((reporting_loss_sum.detach().float(), num_tokens.detach().float()))}
    if router_aux_loss_sum is not None:
        metrics["router aux loss"] = torch.stack((router_aux_loss_sum.detach().float(), num_tokens.detach().float()))
    return loss_sum, num_tokens, metrics


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: torch.nn.Module,
    return_schedule_plan: bool = False,
) -> tuple[torch.Tensor, partial]:
    """Run a fixed-length QwenAir batch with EP-correct loss scaling."""
    if return_schedule_plan:
        raise NotImplementedError("QwenAir does not support the overlapped MoE schedule plan")

    config = get_model_config(model)
    pg_collection = get_pg_collection(model)
    tokens, labels, loss_mask, attention_mask, position_ids, packed = get_batch(
        data_iterator,
        state.cfg,
        False,
        pg_collection=pg_collection,
        vp_stage=None,
    )
    if tokens is None or labels is None or loss_mask is None:
        raise RuntimeError("QwenAir requires tokens, labels, and a loss mask on its only pipeline stage")
    if packed is not None:
        raise NotImplementedError("QwenAir packed-sequence training is not implemented")
    if attention_mask is not None:
        raise NotImplementedError("QwenAir Bridge requires fixed-length batches without an attention mask")

    # Megatron GPT datasets already align each label with the logit at the same
    # position (tokens=text[:-1], labels=text[1:]). Tell the HF-compatible
    # QwenAir model not to shift them a second time.
    valid = loss_mask.bool()
    valid_tokens = valid.sum(dtype=torch.int64)
    masked_labels = torch.where(valid, labels, -100)

    output = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=None,
        labels=masked_labels,
        labels_are_shifted=True,
        output_router_logits=True,
    )
    if not isinstance(output, QwenAirOutput) or output.loss is None:
        raise RuntimeError("QwenAir model must return QwenAirOutput with a training loss")

    # The EP model returns local CE sum / EP-global token count. Recover the
    # numerator before entering MCore's calculate_per_token_loss schedule. For
    # the router objective, qwenair_ep_router_loss divides its differentiable
    # value by EP; multiplying every rank by the EP-global token count makes the
    # final global-token division a token-weighted average across EP replicas.
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


__all__ = ["forward_step", "qwenair_loss"]
