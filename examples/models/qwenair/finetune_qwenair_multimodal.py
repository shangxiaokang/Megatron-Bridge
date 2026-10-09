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

"""Train tiny, randomly initialized QwenAir on real Flickr8k image captions.

The validation target is 128 optimizer steps with global batch size 128 on one
eight-GPU B200/B300 node. The model keeps the canonical QwenAir vocabulary,
special-token IDs, adaptive vision patch geometry, HC/PLE/MoE/QSA structure,
and uses a scaled text/vision width so the integration run completes quickly.
It is a pipeline learning check; its absolute loss is not a proxy for a full
Qwen3.5/QwenAir pretraining curve.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from megatron.bridge.models.qwenair.qwenair_multimodal_step import forward_step
from megatron.bridge.recipes.qwenair.b300.qwenair import (
    qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config,
)
from megatron.bridge.training.pretrain import pretrain


_PROCESSOR_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
_DATASET_REVISION = "81fc5f3a41274c80f17b0406426d57cac57ce6fb"


def parse_args() -> argparse.Namespace:
    """Parse reproducible multimodal convergence-run options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-revision", default=_DATASET_REVISION, help="Immutable tsystems/flickr8k commit SHA")
    parser.add_argument("--processor-revision", default=_PROCESSOR_REVISION)
    parser.add_argument("--train-iters", type=int, default=128)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dataset-split", default="train", help="HF split expression, for example train[:1]")
    parser.add_argument("--indexer-budget", type=int, help="Override the dense-equivalent short-run QSA budget")
    parser.add_argument(
        "--allow-untrained-sparse-indexer",
        action="store_true",
        help="Permit an indexer budget below sequence length for diagnostic A/B runs.",
    )
    parser.add_argument("--lr-decay-style", choices=("constant", "cosine"))
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


def _apply_run_overrides(cfg, args: argparse.Namespace) -> None:
    """Apply diagnostic CLI overrides while keeping the safe short-run defaults."""
    if not args.dataset_split.strip():
        raise ValueError("dataset-split must be non-empty")
    cfg.dataset.source.split = args.dataset_split
    if args.indexer_budget is not None:
        compress_ratio = int(cfg.model.qwenair_text_config["indexer_compress_ratio"])
        if args.indexer_budget < compress_ratio or args.indexer_budget % compress_ratio:
            raise ValueError("indexer-budget must be positive and divisible by indexer_compress_ratio")
        if args.indexer_budget < cfg.dataset.seq_length and not args.allow_untrained_sparse_indexer:
            raise ValueError(
                "indexer-budget below sequence length requires --allow-untrained-sparse-indexer because "
                "the separate QSA indexer training objective is unavailable"
            )
        cfg.model.qwenair_text_config["indexer_budget"] = args.indexer_budget

    if args.lr_decay_style is not None:
        cfg.scheduler.lr_decay_style = args.lr_decay_style
    if args.min_lr is not None:
        if not 0 <= args.min_lr <= cfg.optimizer.lr:
            raise ValueError("min-lr must be in [0, learning rate]")
    effective_style = cfg.scheduler.lr_decay_style
    if effective_style == "constant":
        if args.min_lr is not None and args.min_lr != cfg.optimizer.lr:
            raise ValueError("constant learning rate requires min-lr to equal the learning rate")
        cfg.optimizer.min_lr = cfg.optimizer.lr
    elif args.min_lr is not None:
        cfg.optimizer.min_lr = args.min_lr
    elif args.lr_decay_style == "cosine":
        cfg.optimizer.min_lr = 1.0e-4


def _configure_run_outputs(
    cfg,
    *,
    checkpoint_dir: Path | None,
    tensorboard_dir: Path | None,
) -> None:
    """Use only explicitly requested persistent outputs for diagnostic runs."""
    if checkpoint_dir is None:
        cfg.checkpoint.save = None
        cfg.checkpoint.load = None
        cfg.checkpoint.save_interval = 0
    else:
        cfg.checkpoint.save = str(checkpoint_dir)
        cfg.checkpoint.load = str(checkpoint_dir)
        cfg.checkpoint.save_interval = cfg.train.train_iters
    cfg.logger.tensorboard_dir = None if tensorboard_dir is None else str(tensorboard_dir)


def main() -> None:
    """Build the Flickr8k recipe and launch distributed training."""
    args = parse_args()
    cfg = qwenair_tiny_multimodal_finetune_8gpu_b300_bf16_config(
        dataset_revision=args.dataset_revision,
        processor_revision=args.processor_revision,
        train_iters=args.train_iters,
        global_batch_size=args.global_batch_size,
        image_size=args.image_size,
    )
    _apply_run_overrides(cfg, args)
    _configure_run_outputs(
        cfg,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
    )
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
