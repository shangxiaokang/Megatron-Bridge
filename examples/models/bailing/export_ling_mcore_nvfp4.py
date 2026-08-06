# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Export a Ling MCore NVFP4 checkpoint to sharded Hugging Face weights.

The script preserves the checkpoint's FP32 ``input_scale`` and
``weight_scale_2``, calculates one FP8-E4M3 ``weight_scale`` per 16-weight
block, and packs each quantized weight as NVFP4 in a uint8 tensor.

It supports two scale sources:

* restored ModelOpt quantizers; or
* native MCore/Transformer Engine ``input_scale`` and ``weight_scale_2``
  tensors stored directly in the model state.

Launch this script with ``python -m torch.distributed.run``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict
from unittest import mock

import torch
import yaml
from megatron.core.utils import unwrap_model
from safetensors.torch import save_file
from transformers import AutoTokenizer, PretrainedConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
from megatron.bridge.models.conversion.modelopt_utils import QuantMeta
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


logger = logging.getLogger(__name__)


class _ExportSummary(TypedDict):
    num_shards: int
    num_tensors: int
    total_size: int
    tensor_dtypes: dict[str, torch.dtype]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _is_weight_name(name: str) -> bool:
    return name.endswith(".weight") or name.endswith("_weight")


def _nvfp4_companion_names(weight_name: str) -> tuple[str, str, str]:
    if weight_name.endswith(".weight"):
        base = weight_name.removesuffix(".weight")
        return (
            f"{base}.weight_scale",
            f"{base}.weight_scale_2",
            f"{base}.input_scale",
        )
    if weight_name.endswith("_weight"):
        base = weight_name.removesuffix("_weight")
        return (
            f"{weight_name}_scale",
            f"{weight_name}_scale_2",
            f"{base}_input_scale",
        )
    raise ValueError(f"Not a supported NVFP4 weight name: {weight_name}")


class _ShardedSafetensorsWriter:
    """Write a weight iterator without materializing the full state dict."""

    def __init__(self, output_dir: Path, max_shard_bytes: int) -> None:
        if max_shard_bytes <= 0:
            raise ValueError(f"max_shard_bytes must be positive, got {max_shard_bytes}")
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.total_bytes = 0
        self.temp_shards: list[Path] = []
        self.weight_to_temp_shard: dict[str, str] = {}
        self.tensor_dtypes: dict[str, torch.dtype] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        """Add one tensor and flush the shard when it reaches the size limit."""
        if name in self.tensor_dtypes:
            raise RuntimeError(f"The export yielded duplicate tensor {name!r}")

        tensor = tensor.detach().contiguous().cpu()
        size = _tensor_bytes(tensor)
        if self.buffer and self.buffer_bytes + size > self.max_shard_bytes:
            self._flush()

        self.buffer[name] = tensor
        self.buffer_bytes += size
        self.total_bytes += size
        self.tensor_dtypes[name] = tensor.dtype

    def _flush(self) -> None:
        if not self.buffer:
            return

        shard_number = len(self.temp_shards) + 1
        shard_path = self.output_dir / f".model-{shard_number:05d}.safetensors"
        save_file(self.buffer, str(shard_path), metadata={"format": "pt"})
        for name in self.buffer:
            self.weight_to_temp_shard[name] = shard_path.name
        self.temp_shards.append(shard_path)
        self.buffer = {}
        self.buffer_bytes = 0

    def finish(self) -> _ExportSummary:
        """Finish all shards, rename them, and write the Hugging Face index."""
        self._flush()
        if not self.temp_shards:
            raise RuntimeError("The NVFP4 export did not yield any tensors")

        shard_count = len(self.temp_shards)
        final_names: dict[str, str] = {}
        for shard_number, source_path in enumerate(self.temp_shards, start=1):
            target_name = (
                "model.safetensors"
                if shard_count == 1
                else f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            )
            source_path.rename(self.output_dir / target_name)
            final_names[source_path.name] = target_name

        weight_map = {
            name: final_names[temp_name] for name, temp_name in self.weight_to_temp_shard.items()
        }
        if shard_count > 1:
            index = {
                "metadata": {"total_size": self.total_bytes},
                "weight_map": weight_map,
            }
            with (self.output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as file:
                json.dump(index, file, indent=2, sort_keys=True)
                file.write("\n")

        return {
            "num_shards": shard_count,
            "num_tensors": len(self.tensor_dtypes),
            "total_size": self.total_bytes,
            "tensor_dtypes": self.tensor_dtypes,
        }


def _validate_checkpoint_metadata(checkpoint_dir: Path) -> None:
    """Reject a flat ModelOpt YAML masquerading as a Bridge run config."""
    run_config = checkpoint_dir / "run_config.yaml"
    if not run_config.exists():
        return

    with run_config.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        raise RuntimeError(
            f"{run_config} is not a Megatron-Bridge run config (top-level 'model' is missing). "
            "If it is the previously-created symlink to modelopt_run_config.yaml, remove only "
            "that run_config.yaml symlink. Keep modelopt_run_config.yaml in place."
        )


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory to avoid overwriting a checkpoint."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _patch_ling_config(
    config: PretrainedConfig,
    *,
    physical_vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> None:
    config.vocab_size = physical_vocab_size
    config.num_nextn_predict_layers = 0
    config.first_k_dense_replace = 1
    config.bos_token_id = bos_token_id
    config.eos_token_id = eos_token_id
    config.pad_token_id = pad_token_id


def _scale_name_candidates(param_name: str, suffix: str) -> list[str]:
    """Return possible TE scale-buffer names for one MCore weight."""
    if param_name.endswith(".weight"):
        return [f"{param_name.removesuffix('.weight')}.{suffix}"]

    indexed_weight = re.match(r"^(?P<prefix>.+)\.weight(?P<index>[0-9]+)$", param_name)
    if indexed_weight is None:
        return []

    prefix = indexed_weight.group("prefix")
    expert_index = indexed_weight.group("index")
    candidates = [
        f"{prefix}.{suffix}{expert_index}",
        f"{prefix}.{suffix}",
    ]
    for expert_container in ("experts", "local_experts"):
        marker = f".{expert_container}."
        if marker not in prefix:
            continue
        parent, leaf = prefix.rsplit(marker, maxsplit=1)
        candidates.append(f"{parent}{marker}{expert_index}.{leaf}.{suffix}")
    return candidates


def _find_scale_tensor(
    state: dict[str, torch.Tensor],
    *,
    param_name: str,
    suffix: str,
) -> torch.Tensor | None:
    candidates = _scale_name_candidates(param_name, suffix)
    for candidate in candidates:
        if candidate in state:
            return state[candidate]

    indexed_weight = re.match(r"^(?P<prefix>.+)\.weight(?P<index>[0-9]+)$", param_name)
    if indexed_weight is None:
        return None
    prefix = indexed_weight.group("prefix")
    expert_index = indexed_weight.group("index")
    leaf = prefix.rsplit(".", maxsplit=1)[-1]
    expected_tail = f".{expert_index}.{leaf}.{suffix}"
    matches = [tensor for name, tensor in state.items() if name.endswith(expected_tail)]
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple {suffix} candidates for {param_name}; explicit candidates were {candidates}"
        )
    return matches[0] if matches else None


def _collect_mcore_nvfp4_metadata(
    bridge: AutoBridge,
    model: list[torch.nn.Module],
) -> tuple[list[WeightConversionTask], dict[str, QuantMeta]]:
    """Build ModelOpt export metadata from TE scale buffers saved by MCore."""
    from modelopt.torch.export.quant_utils import QUANTIZATION_NVFP4

    conversion_tasks = bridge._model_bridge.build_conversion_tasks(bridge.hf_pretrained, model)
    state_cache: dict[int, dict[str, torch.Tensor]] = {}
    metadata: dict[str, QuantMeta] = {}

    for task in conversion_tasks:
        if task.megatron_module is None or task.param_weight is None:
            continue
        module_id = id(task.megatron_module)
        if module_id not in state_cache:
            state_cache[module_id] = {
                name: value
                for name, value in task.megatron_module.state_dict(keep_vars=True).items()
                if isinstance(value, torch.Tensor)
            }
        state = state_cache[module_id]
        input_scale = _find_scale_tensor(state, param_name=task.param_name, suffix="input_scale")
        weight_scale_2 = _find_scale_tensor(state, param_name=task.param_name, suffix="weight_scale_2")

        if input_scale is None and weight_scale_2 is None:
            continue
        if input_scale is None or weight_scale_2 is None:
            raise RuntimeError(
                f"Incomplete NVFP4 scale state for {task.param_name}: "
                f"input_scale={input_scale is not None}, weight_scale_2={weight_scale_2 is not None}"
            )

        input_scale = input_scale.detach().float().abs().cpu()
        weight_scale_2 = weight_scale_2.detach().float().abs().cpu()
        if not torch.isfinite(input_scale).all() or not torch.all(input_scale > 0):
            raise RuntimeError(f"Invalid input_scale for {task.param_name}: {input_scale}")
        if not torch.isfinite(weight_scale_2).all() or not torch.all(weight_scale_2 > 0):
            raise RuntimeError(f"Invalid weight_scale_2 for {task.param_name}: {weight_scale_2}")

        # MCore/TE stores deployment scales, while QuantMeta carries amax. Multiplying
        # input_scale by 6 * 448 lets the canonical exporter reproduce it exactly.
        metadata[task.global_param_name] = QuantMeta(
            qformat=QUANTIZATION_NVFP4,
            block_size=16,
            weight_amax=weight_scale_2 * (6.0 * 448.0),
            weight_scale_2=weight_scale_2,
            input_amax=input_scale * (6.0 * 448.0),
        )

    if not metadata:
        raise RuntimeError(
            "No ModelOpt quantizers and no MCore input_scale/weight_scale_2 tensors were found in the loaded model"
        )
    return conversion_tasks, metadata


def _export_nvfp4_weights(
    bridge: AutoBridge,
    model: list[torch.nn.Module],
    *,
    show_progress: bool,
) -> Iterator[HFWeightTuple]:
    """Yield NVFP4 HF tensors from ModelOpt state or native MCore TE scales."""
    from modelopt.torch.quantization.utils import is_quantized

    model_parts = unwrap_model(model)
    if any(is_quantized(part) for part in model_parts):
        if show_progress:
            logger.info("Using restored ModelOpt quantizer metadata for NVFP4 export")
        yield from bridge.export_hf_weights_modelopt(
            model,
            quant_mode="nvfp4",
            cpu=False,
            show_progress=show_progress,
        )
        return

    from megatron.bridge.models.conversion import modelopt_utils

    conversion_tasks, metadata = _collect_mcore_nvfp4_metadata(bridge, model)
    if show_progress:
        logger.info("Using %s MCore/TE input_scale + weight_scale_2 records", len(metadata))
    with mock.patch.object(modelopt_utils, "collect_modelopt_quant_metadata", return_value=metadata):
        yield from bridge.export_hf_weights_modelopt(
            model,
            quant_mode="nvfp4",
            cpu=False,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
        )


def _validate_exported_nvfp4_tensors(tensor_dtypes: dict[str, torch.dtype]) -> tuple[int, list[str]]:
    quantized_weights = [
        name for name, dtype in tensor_dtypes.items() if _is_weight_name(name) and dtype == torch.uint8
    ]
    if not quantized_weights:
        raise RuntimeError("No packed uint8 NVFP4 weights were exported")

    missing: list[str] = []
    wrong_dtype: list[str] = []
    for weight_name in quantized_weights:
        weight_scale, weight_scale_2, input_scale = _nvfp4_companion_names(weight_name)
        for companion in (weight_scale, weight_scale_2, input_scale):
            if companion not in tensor_dtypes:
                missing.append(f"{weight_name} -> {companion}")
        if tensor_dtypes.get(weight_scale) != torch.float8_e4m3fn:
            wrong_dtype.append(f"{weight_scale}: {tensor_dtypes.get(weight_scale)}")
        if tensor_dtypes.get(weight_scale_2) != torch.float32:
            wrong_dtype.append(f"{weight_scale_2}: {tensor_dtypes.get(weight_scale_2)}")
        if tensor_dtypes.get(input_scale) != torch.float32:
            wrong_dtype.append(f"{input_scale}: {tensor_dtypes.get(input_scale)}")

    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Missing NVFP4 companion tensors ({len(missing)}):\n{preview}")
    if wrong_dtype:
        preview = "\n".join(wrong_dtype[:20])
        raise RuntimeError(f"Unexpected NVFP4 scale dtypes ({len(wrong_dtype)}):\n{preview}")

    excluded_modules = sorted(
        name.removesuffix(".weight")
        for name, dtype in tensor_dtypes.items()
        if name.endswith(".weight") and dtype != torch.uint8
    )
    return len(quantized_weights), excluded_modules


def _save_hf_artifacts(
    bridge: AutoBridge,
    output_dir: Path,
    *,
    hf_model_id: str,
    tokenizer_id: str,
    tokenizer_vocab_size: int,
    physical_vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    excluded_modules: list[str],
    trust_remote_code: bool,
) -> None:
    additional_files = getattr(bridge._model_bridge, "ADDITIONAL_FILE_PATTERNS", None) or None
    bridge.hf_pretrained.save_artifacts(
        output_dir,
        original_source_path=hf_model_id,
        additional_files=additional_files,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
    if len(tokenizer) != tokenizer_vocab_size:
        raise RuntimeError(
            f"Tokenizer size mismatch: expected {tokenizer_vocab_size}, got {len(tokenizer)} from {tokenizer_id}"
        )
    expected_token_ids = {
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    actual_token_ids = {name: getattr(tokenizer, name) for name in expected_token_ids}
    if actual_token_ids != expected_token_ids:
        raise RuntimeError(f"Tokenizer special-token mismatch: expected {expected_token_ids}, got {actual_token_ids}")
    tokenizer.save_pretrained(output_dir)

    config_path = output_dir / "config.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    config.update(
        {
            "vocab_size": physical_vocab_size,
            "num_nextn_predict_layers": 0,
            "first_k_dense_replace": 1,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
    )
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")

    generation_config_path = output_dir / "generation_config.json"
    if generation_config_path.exists():
        with generation_config_path.open(encoding="utf-8") as file:
            generation_config = json.load(file)
        generation_config.update(expected_token_ids)
        with generation_config_path.open("w", encoding="utf-8") as file:
            json.dump(generation_config, file, indent=2, sort_keys=True)
            file.write("\n")

    try:
        modelopt_version = importlib.metadata.version("nvidia-modelopt")
    except importlib.metadata.PackageNotFoundError:
        modelopt_version = "unknown"
    hf_quant_config = {
        "producer": {"name": "modelopt", "version": modelopt_version},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": excluded_modules,
        },
    }
    with (output_dir / "hf_quant_config.json").open("w", encoding="utf-8") as file:
        json.dump(hf_quant_config, file, indent=2, sort_keys=True)
        file.write("\n")


@torchrun_main
def main(
    hf_model_id: str,
    tokenizer_id: str,
    checkpoint_dir: str,
    output_dir: str,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    max_shard_size_gib: float,
    physical_vocab_size: int,
    tokenizer_vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    trust_remote_code: bool,
) -> None:
    """Load the distributed checkpoint and export a complete NVFP4 HF directory."""
    if os.environ.get("WORLD_SIZE") is None:
        raise RuntimeError("Launch this script with python -m torch.distributed.run")

    expected_world_size = tp * pp * ep
    actual_world_size = torch.distributed.get_world_size()
    if actual_world_size != expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={actual_world_size}, but TP*PP*EP={tp}*{pp}*{ep}={expected_world_size}"
        )

    checkpoint_path = Path(checkpoint_dir).resolve()
    output_path = Path(output_dir).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    _validate_checkpoint_metadata(checkpoint_path)

    rank = torch.distributed.get_rank()
    if rank == 0:
        _prepare_output_dir(output_path)
    torch.distributed.barrier()

    trusted = is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model_id)
    bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=trusted)
    _patch_ling_config(
        bridge.hf_pretrained.config,
        physical_vocab_size=physical_vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.pipeline_dtype = torch.bfloat16
    provider.finalize()
    provider.initialize_model_parallel(seed=0)

    model = bridge.load_megatron_model(
        checkpoint_path,
        mp_overrides={
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
        },
        wrap_with_ddp=False,
    )

    writer = (
        _ShardedSafetensorsWriter(
            output_path,
            max_shard_bytes=int(max_shard_size_gib * 1024**3),
        )
        if rank == 0
        else None
    )
    for name, tensor in _export_nvfp4_weights(bridge, model, show_progress=rank == 0):
        if writer is not None:
            writer.add(name, tensor)

    if writer is not None:
        summary = writer.finish()
        quantized_count, excluded_modules = _validate_exported_nvfp4_tensors(summary["tensor_dtypes"])
        _save_hf_artifacts(
            bridge,
            output_path,
            hf_model_id=hf_model_id,
            tokenizer_id=tokenizer_id,
            tokenizer_vocab_size=tokenizer_vocab_size,
            physical_vocab_size=physical_vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            excluded_modules=excluded_modules,
            trust_remote_code=trusted,
        )
        logger.info(
            "Export complete: %s tensors, %s packed NVFP4 weights, %s shards, %.2f GiB at %s",
            summary["num_tensors"],
            quantized_count,
            summary["num_shards"],
            summary["total_size"] / 1024**3,
            output_path,
        )
    torch.distributed.barrier()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-id", default="inclusionAI/Ling-mini-base-2.0")
    parser.add_argument("--tokenizer-id", default="moonshotai/Moonlight-16B-A3B-Instruct")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--max-shard-size-gib", type=float, default=4.0)
    parser.add_argument("--physical-vocab-size", type=int, default=163968)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=163842)
    parser.add_argument("--bos-token-id", type=int, default=163584)
    parser.add_argument("--eos-token-id", type=int, default=163585)
    parser.add_argument("--pad-token-id", type=int, default=163838)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    main(
        args.hf_model_id,
        args.tokenizer_id,
        args.checkpoint_dir,
        args.output_dir,
        args.tp,
        args.pp,
        args.ep,
        args.etp,
        args.max_shard_size_gib,
        args.physical_vocab_size,
        args.tokenizer_vocab_size,
        args.bos_token_id,
        args.eos_token_id,
        args.pad_token_id,
        args.trust_remote_code,
    )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
