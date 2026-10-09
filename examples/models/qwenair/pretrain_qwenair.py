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

"""Run QwenAir LM training through Megatron Bridge.

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
    configure_qwenair_indexed_data,
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
    parser.add_argument(
        "--seq-length",
        type=int,
        default=64,
        help="Target-recipe sequence length (64-token conservative bring-up default)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Megatron indexed-dataset prefix, without .bin or .idx; omit for mock data",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        help="From-scratch indexed-data vocabulary override; requires --tokenizer-eod-id",
    )
    parser.add_argument(
        "--tokenizer-eod-id",
        type=int,
        help="From-scratch indexed-data EOD override; requires --tokenizer-vocab-size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Peak learning rate for a bounded indexed-data run; requires --min-learning-rate",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        help="Minimum learning rate for a bounded indexed-data run; requires --learning-rate",
    )
    parser.add_argument("--train-iters", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


def _validate_indexed_dataset_prefix(data_path: Path) -> None:
    """Require both files that form a Megatron indexed dataset."""
    missing = [
        str(Path(f"{data_path}{suffix}")) for suffix in (".bin", ".idx") if not Path(f"{data_path}{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"indexed dataset is incomplete; missing: {', '.join(missing)}")


def main() -> None:
    """Build the selected recipe and enter the Bridge training loop."""
    args = parse_args()
    if args.train_iters is not None and args.train_iters < 1:
        raise ValueError("train-iters must be positive")
    token_contract_override = (args.tokenizer_vocab_size, args.tokenizer_eod_id)
    if (args.tokenizer_vocab_size is None) != (args.tokenizer_eod_id is None):
        raise ValueError("tokenizer-vocab-size and tokenizer-eod-id must be provided together")
    learning_rate_override = (args.learning_rate, args.min_learning_rate)
    if (args.learning_rate is None) != (args.min_learning_rate is None):
        raise ValueError("learning-rate and min-learning-rate must be provided together")
    if args.data_path is None and any(
        value is not None for value in (*token_contract_override, *learning_rate_override)
    ):
        raise ValueError("tokenizer and learning-rate overrides require data-path")
    if args.data_path is not None:
        _validate_indexed_dataset_prefix(args.data_path)

    if args.target_config is None:
        cfg = qwenair_tiny_pretrain_8gpu_b300_bf16_config()
    else:
        cfg = qwenair_text_pretrain_32gpu_b300_bf16_config(
            args.target_config,
            seq_length=args.seq_length,
        )
    if args.data_path is not None:
        real_data_train_iters = args.train_iters if args.train_iters is not None else 100
        cfg = configure_qwenair_indexed_data(
            cfg,
            args.data_path,
            train_iters=real_data_train_iters,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
            tokenizer_eod_id=args.tokenizer_eod_id,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
        )
    if args.train_iters is not None:
        cfg.train.train_iters = args.train_iters
        cfg.checkpoint.save_interval = args.train_iters
    if args.checkpoint_dir is not None:
        cfg.checkpoint.save = str(args.checkpoint_dir)
        cfg.checkpoint.load = str(args.checkpoint_dir)
    if args.tensorboard_dir is not None:
        cfg.logger.tensorboard_dir = str(args.tensorboard_dir)
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
