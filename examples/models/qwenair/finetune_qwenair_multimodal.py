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

"""Train tiny QwenAir end to end on the real Flickr8k image-caption dataset.

The validation target is 128 optimizer steps with global batch size 128 on one
eight-GPU B200/B300 node. The model keeps the canonical QwenAir vocabulary,
special-token IDs, adaptive vision patch geometry, HC/PLE/MoE/QSA structure,
and uses a scaled text/vision width so the integration run completes quickly.
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
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


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
    if args.checkpoint_dir is not None:
        cfg.checkpoint.save = str(args.checkpoint_dir)
        cfg.checkpoint.load = str(args.checkpoint_dir)
    if args.tensorboard_dir is not None:
        cfg.logger.tensorboard_dir = str(args.tensorboard_dir)
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
