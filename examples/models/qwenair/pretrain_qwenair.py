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

"""Run QwenAir mock-data LM training through Megatron Bridge.

Eight-GPU integration example::

    uv run python -m torch.distributed.run --nproc_per_node=8 \
        examples/models/qwenair/pretrain_qwenair.py

The target recipe needs four eight-GPU nodes and the pinned model JSON. Launch
one task per node and let ``torch.distributed.run`` start the eight local ranks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from megatron.bridge.models.qwenair.qwenair_step import forward_step
from megatron.bridge.recipes.qwenair.b300.qwenair import (
    qwenair_text_pretrain_32gpu_b300_bf16_config,
    qwenair_tiny_pretrain_8gpu_b300_bf16_config,
)
from megatron.bridge.training.pretrain import pretrain


def parse_args() -> argparse.Namespace:
    """Parse the bounded recipe and output overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-config",
        type=Path,
        help="Use the 32-GPU target-text recipe with this QwenAir config JSON",
    )
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--train-iters", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    """Build the selected recipe and enter the Bridge training loop."""
    args = parse_args()
    if args.target_config is None:
        cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()
    else:
        cfg = qwenair_text_pretrain_32gpu_b300_bf16_config(
            args.target_config,
            seq_length=args.seq_length,
        )
    if args.train_iters is not None:
        if args.train_iters < 1:
            raise ValueError("train-iters must be positive")
        cfg.train.train_iters = args.train_iters
        cfg.checkpoint.save_interval = args.train_iters
    if args.checkpoint_dir is not None:
        cfg.checkpoint.save = str(args.checkpoint_dir)
        cfg.checkpoint.load = str(args.checkpoint_dir)
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
