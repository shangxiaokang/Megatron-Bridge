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

"""Build a bounded QwenAir indexed dataset from real-text Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoTokenizer


_QWENAIR_VOCAB_SIZE = 248_320
_QWENAIR_EOD_ID = 248_044
_LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse deterministic extraction and preprocessing options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--vocab-size", type=int, default=_QWENAIR_VOCAB_SIZE)
    parser.add_argument("--eod-token-id", type=int, default=_QWENAIR_EOD_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="qwenair-real-text")
    parser.add_argument("--max-documents", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-revision")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_tokenizer(
    model: str,
    revision: str | None,
    output_dir: Path,
    *,
    vocab_size: int,
    eod_token_id: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(f"tokenizer output already exists; pass --overwrite to replace: {output_dir}")
        non_files = [str(path) for path in existing if not path.is_file()]
        if non_files:
            raise FileExistsError(f"refusing to replace non-file tokenizer output: {', '.join(non_files)}")
        for path in existing:
            path.unlink()
    kwargs = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
    if len(tokenizer) != vocab_size:
        raise ValueError(f"tokenizer vocabulary must be {vocab_size}, got {len(tokenizer)}")

    eod_token = tokenizer.convert_ids_to_tokens(eod_token_id)
    if tokenizer.convert_tokens_to_ids(eod_token) != eod_token_id:
        raise ValueError(f"tokenizer does not contain EOD id {eod_token_id}")
    tokenizer.eos_token = eod_token
    if tokenizer.eos_token_id != eod_token_id or len(tokenizer) != vocab_size:
        raise ValueError("setting the EOD token changed the tokenizer contract")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    tokenizer_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    return {
        "source": model,
        "requested_revision": revision,
        "materialized_path": str(output_dir),
        "vocab_size": len(tokenizer),
        "eod_token": eod_token,
        "eod_token_id": tokenizer.eos_token_id,
        "files": tokenizer_files,
    }


def _write_jsonl(
    parquet_paths: list[Path],
    output_path: Path,
    *,
    text_column: str,
    max_documents: int,
) -> tuple[int, int, str]:
    documents = 0
    characters = 0
    digest = hashlib.sha256()
    with output_path.open("wb") as output_file:
        for parquet_path in parquet_paths:
            parquet_file = pq.ParquetFile(parquet_path)
            if text_column not in parquet_file.schema_arrow.names:
                raise ValueError(f"column {text_column!r} is missing from {parquet_path}")
            for batch in parquet_file.iter_batches(columns=[text_column], batch_size=256):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text, str) or not text.strip():
                        continue
                    encoded = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
                    output_file.write(encoded)
                    digest.update(encoded)
                    documents += 1
                    characters += len(text)
                    if documents == max_documents:
                        return documents, characters, digest.hexdigest()
    return documents, characters, digest.hexdigest()


def _remove_existing_outputs(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists; pass --overwrite to replace: {joined}")
    for path in existing:
        path.unlink()


def _remove_dataset_cache(cache_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    if not cache_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"dataset index cache already exists; pass --overwrite to replace: {cache_dir}")
    resolved_cache = cache_dir.resolve()
    resolved_cache.relative_to(output_dir.resolve())
    shutil.rmtree(resolved_cache)


def _inspect_indexed_dataset(dataset_prefix: Path, *, vocab_size: int, eod_token_id: int) -> dict[str, int]:
    dataset = IndexedDataset(str(dataset_prefix))
    token_count = 0
    eod_count = 0
    max_token_id = -1
    documents_ending_in_eod = 0
    for index in range(len(dataset)):
        tokens = dataset[index]
        if len(tokens) == 0:
            continue
        token_count += len(tokens)
        eod_count += int((tokens == eod_token_id).sum())
        max_token_id = max(max_token_id, int(tokens.max()))
        documents_ending_in_eod += int(tokens[-1] == eod_token_id)
    if max_token_id >= vocab_size:
        raise ValueError(f"indexed token {max_token_id} exceeds configured vocabulary")
    if eod_count == 0 or documents_ending_in_eod != len(dataset):
        raise ValueError("indexed documents do not preserve the QwenAir EOD boundary")
    return {
        "documents": len(dataset),
        "tokens": token_count,
        "eod_tokens": eod_count,
        "documents_ending_in_eod": documents_ending_in_eod,
        "max_token_id": max_token_id,
    }


def main() -> None:
    """Extract text, pin the tokenizer, and invoke MCore preprocessing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.max_documents < 1:
        raise ValueError("max-documents must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.vocab_size < 1 or not 0 <= args.eod_token_id < args.vocab_size:
        raise ValueError("eod-token-id must be in [0, vocab-size)")
    missing_inputs = [str(path) for path in args.input_parquet if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"missing parquet input: {', '.join(missing_inputs)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.output_name}.jsonl"
    output_prefix = args.output_dir / args.output_name
    dataset_prefix = Path(f"{output_prefix}_text_document")
    manifest_path = args.output_dir / f"{args.output_name}.manifest.json"
    output_paths = [jsonl_path, Path(f"{dataset_prefix}.bin"), Path(f"{dataset_prefix}.idx"), manifest_path]
    _remove_existing_outputs(output_paths, overwrite=args.overwrite)
    _remove_dataset_cache(dataset_prefix / "cache", args.output_dir, overwrite=args.overwrite)

    tokenizer_dir = args.output_dir / f"{args.output_name}-tokenizer"
    tokenizer_metadata = _prepare_tokenizer(
        args.tokenizer_model,
        args.tokenizer_revision,
        tokenizer_dir,
        vocab_size=args.vocab_size,
        eod_token_id=args.eod_token_id,
        overwrite=args.overwrite,
    )
    documents, characters, jsonl_sha256 = _write_jsonl(
        args.input_parquet,
        jsonl_path,
        text_column=args.text_column,
        max_documents=args.max_documents,
    )
    if documents < args.max_documents:
        raise ValueError(f"only found {documents} non-empty documents; requested {args.max_documents}")

    repo_root = Path(__file__).resolve().parents[3]
    preprocess_script = repo_root / "3rdparty" / "Megatron-LM" / "tools" / "preprocess_data.py"
    command = [
        sys.executable,
        str(preprocess_script),
        "--input",
        str(jsonl_path),
        "--output-prefix",
        str(output_prefix),
        "--json-keys",
        "text",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(tokenizer_dir),
        "--append-eod",
        "--workers",
        str(args.workers),
    ]
    _LOGGER.info("Running Megatron preprocessing for %d documents", documents)
    subprocess.run(command, cwd=repo_root, check=True)

    indexed_files = [Path(f"{dataset_prefix}.bin"), Path(f"{dataset_prefix}.idx")]
    if not all(path.is_file() and path.stat().st_size > 0 for path in indexed_files):
        raise RuntimeError("Megatron preprocessing did not produce a complete indexed dataset")
    indexed_statistics = _inspect_indexed_dataset(
        dataset_prefix,
        vocab_size=args.vocab_size,
        eod_token_id=args.eod_token_id,
    )
    manifest = {
        "format": "qwenair-real-text/v1",
        "source": {
            "parquet_files": [{"path": str(path), "bytes": path.stat().st_size} for path in args.input_parquet],
            "revision": args.source_revision,
            "text_column": args.text_column,
        },
        "selection": {"method": "first-non-empty-in-file-order", "documents": documents},
        "jsonl": {"path": str(jsonl_path), "characters": characters, "sha256": jsonl_sha256},
        "tokenizer": tokenizer_metadata,
        "indexed_dataset": {
            "prefix": str(dataset_prefix),
            "statistics": indexed_statistics,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)} for path in indexed_files
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _LOGGER.info("Indexed dataset prefix: %s", dataset_prefix)
    _LOGGER.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
