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

"""Train canonical full-geometry QwenAir on Flickr8k for 1024 steps.

The default run uses 32 B300 ranks, EP32, BF16, sequence length 128, global
batch size 128, and a 64-step warmup into cosine decay from 3e-4 to 3e-5.
Persistent outputs are disabled unless their directories are supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from megatron.bridge.models.qwenair.qwenair_multimodal_step import forward_step
from megatron.bridge.recipes.qwenair.b300 import (
    qwenair_target_multimodal_finetune_32gpu_b300_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain


_PROCESSOR_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
_DATASET_REVISION = "81fc5f3a41274c80f17b0406426d57cac57ce6fb"
_PROCESSOR_PATH = "Qwen/Qwen3.5-0.8B"
_CANONICAL_MODEL_VOCAB_SIZE = 248_320
_PROCESSOR_SPECIAL_TOKEN_IDS = {
    "<|endoftext|>": 248_044,
    "<|im_end|>": 248_046,
    "<|vision_start|>": 248_053,
    "<|vision_end|>": 248_054,
    "<|image_pad|>": 248_056,
    "<|video_pad|>": 248_057,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical target-run options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to configs-and-numbers provenance/bf16-model-config.json",
    )
    parser.add_argument("--dataset-revision", default=_DATASET_REVISION, help="Immutable tsystems/flickr8k commit SHA")
    parser.add_argument("--processor-revision", default=_PROCESSOR_REVISION)
    parser.add_argument("--train-iters", type=int, default=1024)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dataset-split", default="train", help="HF split expression, for example train[:1]")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args(argv)


def _configure_run_outputs(
    config: ConfigContainer,
    *,
    checkpoint_dir: Path | None,
    tensorboard_dir: Path | None,
) -> None:
    """Enable only explicitly requested persistent outputs."""
    if checkpoint_dir is None:
        config.checkpoint.save = None
        config.checkpoint.load = None
        config.checkpoint.save_interval = 0
    else:
        config.checkpoint.save = str(checkpoint_dir)
        config.checkpoint.load = str(checkpoint_dir)
        config.checkpoint.save_interval = config.train.train_iters
    config.logger.tensorboard_dir = None if tensorboard_dir is None else str(tensorboard_dir)


def _validate_processor_contract(processor: object, *, model_vocab_size: int) -> None:
    """Fail if a loaded processor does not implement QwenAir token semantics."""
    tokenizer = getattr(processor, "tokenizer", processor)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert_tokens_to_ids):
        raise TypeError("QwenAir processor tokenizer must implement convert_tokens_to_ids")
    try:
        observed = {
            "tokenizer_size": len(tokenizer),
            "eos_token_id": int(getattr(tokenizer, "eos_token_id")),
            **{token: int(convert_tokens_to_ids(token)) for token in _PROCESSOR_SPECIAL_TOKEN_IDS},
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("QwenAir processor exposes invalid vocabulary or special-token IDs") from exc
    expected = {
        # Qwen3.5 uses <|im_end|> as its chat tokenizer EOS.  The canonical
        # model EOS remains <|endoftext|>, which is checked by the mapping.
        "eos_token_id": _PROCESSOR_SPECIAL_TOKEN_IDS["<|im_end|>"],
        **_PROCESSOR_SPECIAL_TOKEN_IDS,
    }
    mismatches = [name for name, value in expected.items() if observed[name] != value]
    if int(model_vocab_size) != _CANONICAL_MODEL_VOCAB_SIZE:
        mismatches.append("model_vocab_size")
        observed["model_vocab_size"] = int(model_vocab_size)
        expected["model_vocab_size"] = _CANONICAL_MODEL_VOCAB_SIZE
    if observed["tokenizer_size"] > int(model_vocab_size):
        mismatches.append("tokenizer_size")
        expected["tokenizer_size"] = int(model_vocab_size)
    if mismatches:
        details = ", ".join(f"{name}={observed[name]} (expected {expected[name]})" for name in mismatches)
        raise ValueError(f"QwenAir processor token contract mismatch: {details}")


def _validate_processor_revision(processor_revision: str, *, model_vocab_size: int) -> None:
    """Load the pinned processor revision and validate its token contract."""
    from transformers import AutoConfig, AutoProcessor

    processor_config = AutoConfig.from_pretrained(_PROCESSOR_PATH, revision=processor_revision)
    processor_text_config = getattr(processor_config, "text_config", processor_config)
    try:
        processor_vocab_size = int(getattr(processor_text_config, "vocab_size"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("QwenAir processor source config does not declare text vocabulary capacity") from exc
    if processor_vocab_size != int(model_vocab_size):
        raise ValueError(
            "QwenAir processor source config vocabulary mismatch: "
            f"vocab_size={processor_vocab_size} (expected {model_vocab_size})"
        )

    processor = AutoProcessor.from_pretrained(_PROCESSOR_PATH, revision=processor_revision)
    _validate_processor_contract(processor, model_vocab_size=model_vocab_size)


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Build the target recipe and apply non-architectural CLI controls."""
    config = qwenair_target_multimodal_finetune_32gpu_b300_bf16_config(
        args.model_config,
        dataset_revision=args.dataset_revision,
        processor_revision=args.processor_revision,
        train_iters=args.train_iters,
        global_batch_size=args.global_batch_size,
        seq_length=args.seq_length,
        image_size=args.image_size,
    )
    if not args.dataset_split.strip():
        raise ValueError("dataset_split must be non-empty")
    config.dataset.source.split = args.dataset_split
    _configure_run_outputs(
        config,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
    )
    return config


def main() -> None:
    """Launch canonical QwenAir multimodal training."""
    args = parse_args()
    config = build_config(args)
    _validate_processor_revision(args.processor_revision, model_vocab_size=int(config.model.vocab_size))
    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
