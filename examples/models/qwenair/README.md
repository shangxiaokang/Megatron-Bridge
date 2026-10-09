# QwenAir training

`pretrain_qwenair.py` supports the two-step mock-data smoke recipe and a bounded real-data run. Real text must first be converted to a Megatron indexed dataset with the QwenAir token contract:

- vocabulary size: `248320`
- EOD token ID: `248044`

The EOD value is part of the model semantics because QwenAir PLE uses it to prevent n-grams from crossing document boundaries.

Prepare a deterministic sample from one or more real-text Parquet files. The tokenizer source may be a local snapshot or a Hugging Face model with the QwenAir-compatible 248320-token vocabulary. The script materializes the tokenizer, maps its EOS to the QwenAir EOD token, validates both IDs, and writes a provenance manifest.

```bash
uv run python examples/models/qwenair/prepare_real_text.py \
  --input-parquet /shared/data/openwebtext/00000.parquet \
  --tokenizer-model Qwen/Qwen3.5-0.8B \
  --tokenizer-revision <immutable-commit> \
  --source-revision <immutable-dataset-commit> \
  --output-dir /shared/data/qwenair-openwebtext \
  --max-documents 4096 \
  --workers 16
```

The training path is the generated prefix ending in `_text_document`, without `.bin` or `.idx`:

```bash
uv run python -m torch.distributed.run --nproc-per-node=8 \
  examples/models/qwenair/pretrain_qwenair.py \
  --data-path /shared/data/qwenair-openwebtext/qwenair-real-text_text_document \
  --train-iters 100 \
  --checkpoint-dir /shared/checkpoints/qwenair-real-100 \
  --tensorboard-dir /shared/logs/qwenair-real-100/tensorboard
```

The real-data configuration uses EP4 x EDP2, BF16, a global batch of 8, sequence length 64, 10 warmup steps, and cosine decay through step 100. Use a fresh checkpoint directory when changing the tokenizer or vocabulary.

For an offline from-scratch convergence test, an existing tokenizer can be used with an explicit contract. For example, GPT-2 tokenized OpenWebText uses vocabulary size `50257` and EOD `50256`:

```bash
uv run python examples/models/qwenair/prepare_real_text.py \
  --input-parquet /shared/data/openwebtext/00000.parquet \
  --tokenizer-model /shared/tokenizers/gpt2 \
  --vocab-size 50257 \
  --eod-token-id 50256 \
  --output-dir /shared/data/qwenair-openwebtext

uv run python -m torch.distributed.run --nproc-per-node=8 \
  examples/models/qwenair/pretrain_qwenair.py \
  --data-path /shared/data/qwenair-openwebtext/qwenair-real-text_text_document \
  --tokenizer-vocab-size 50257 \
  --tokenizer-eod-id 50256 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --train-iters 100
```

This override validates QwenAir training from random initialization. The explicit learning rates are suitable for the 3.4M-parameter tiny recipe's bounded convergence check; the recipe defaults remain available for longer runs. Keep the default 248320/248044 token contract for QwenAir checkpoint compatibility.

After training, extract the per-step curve and convergence verdict from the Slurm log:

```bash
uv run python examples/models/qwenair/analyze_training_log.py \
  --log /shared/logs/qwenair-real-100.out \
  --output-dir /shared/logs/qwenair-real-100-analysis \
  --expected-steps 100 \
  --warmup-steps 10 \
  --global-batch-size 8
```

The analyzer requires all 100 steps, finite metrics, and zero skipped/NaN iterations. It compares steps 11–20 with 91–100 and fits a post-warmup linear trend. By default, `PASS` means only that these numerical-health and downward-trend checks succeeded: at least a 2% mean loss reduction, a negative slope, and at least seven of the final ten points below the early-window median. Use `--max-final-loss` when a workload has a separately justified absolute acceptance target.

## Real multimodal pipeline-learning run

`finetune_qwenair_multimodal.py` trains the native Transformer Engine vision encoder and the QwenAir language model together. The reproducible recipe pins:

- `tsystems/flickr8k` at revision `81fc5f3a41274c80f17b0406426d57cac57ce6fb`
- `Qwen/Qwen3.5-0.8B` processor at revision `2fc06364715b967f1860aea9cf38778875588b17`
- QwenAir image/video token IDs `248056` and `248057`
- patch size 16, temporal patch size 2, and spatial merge size 2

Flickr8k contains 8,091 images with five captions each. The adapter expands these into 40,455 image-caption pairs, then applies a seed-1234 shuffle before selecting the 16,384 samples consumed by 128 steps at global batch size 128. This selection covers 7,499 distinct images. The `224` image option is an equal minimum/maximum pixel budget; the processor preserves aspect ratio, so each sample's grid and visual-token count remain dynamic.

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/models/qwenair/finetune_qwenair_multimodal.py \
  --train-iters 128 \
  --global-batch-size 128 \
  --image-size 224 \
  --checkpoint-dir /shared/checkpoints/qwenair-flickr8k-128 \
  --tensorboard-dir /shared/logs/qwenair-flickr8k-128/tensorboard
```

The bounded recipe uses BF16, EP4 x EDP2, micro batch size 1, sequence length 128, 12 warmup steps, and a constant `1e-3` learning rate after warmup. Its QSA token budget is 128, which is dense-equivalent at this sequence length and prevents the unavailable hard-top-k indexer objective from leaving a random sparse selector in the language-model path. It preserves the QwenAir token, PLE, HC, MoE, QSA-kernel, and multimodal scatter contracts while reducing the text and vision widths for an integration test. MTP and sparse-indexer training remain outside this run, so its absolute loss must not be compared with a full Qwen3.5/QwenAir pretraining curve.

For a deterministic pipeline-learning diagnostic, repeat one Flickr8k image and its five captions. This is an intentional overfit test, not a generalization measurement:

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/models/qwenair/finetune_qwenair_multimodal.py \
  --dataset-split 'train[:1]' \
  --train-iters 128 \
  --global-batch-size 128
```

Analyze the run with its exact schedule and batch contract:

```bash
uv run python examples/models/qwenair/analyze_training_log.py \
  --log /shared/logs/qwenair-flickr8k-128.out \
  --output-dir /shared/logs/qwenair-flickr8k-128-analysis \
  --expected-steps 128 \
  --warmup-steps 12 \
  --global-batch-size 128 \
  --require-pass
```

The analyzer also checks every step's consumed-sample count. Add a workload-specific target such as `--max-final-loss <value>` before using `--require-pass` as an absolute convergence gate; without it, `PASS` reports numerical health and a downward trend only.

## Canonical full-geometry multimodal diagnostic

`finetune_qwenair_target_multimodal.py` is the 32-B300 recipe for the
`configs-and-numbers` provenance config. It reads the JSON at runtime and
fails closed unless it describes the full `qwen4_exp` conditional-generation
model with `language_model_only=false`, the 48-layer text geometry, the
27-layer vision encoder, and the canonical text and vision token IDs. It does
not accept the language-only `qwen3_8_flash_next` alias.

The default training contract is EP32, BF16, sequence length 128, global batch
size 128, micro batch size 1, and 1024 optimizer steps. The learning rate warms
up for 64 steps to `1e-4`, then decays with a cosine schedule to `1e-5` at step
1024. The run consumes 131,072 image-caption pairs, approximately 3.24 passes
over the 40,455 Flickr8k pairs, so it is an optimizer and numerical-health
diagnostic rather than a pretraining-quality measurement.

Launch 32 ranks with Slurm. `slurm_target_multimodal.sh` requests four
eight-GPU nodes and starts one Slurm task per GPU. It invokes the configured
Python executable directly; do not put `torchrun` around it. The launcher
maps `SLURM_PROCID`, `SLURM_NTASKS`, and `SLURM_LOCALID` to `RANK`,
`WORLD_SIZE`, and `LOCAL_RANK`, and derives a common rendezvous address and
job-specific port before starting Python. At worker startup it queries
`torch.cuda.device_count()`: `LOCAL_RANK` is the Slurm local task ID when all
node GPUs are visible, and zero when Slurm/Pyxis exposes only the one GPU
assigned to that task.

Use clean, immutable checkouts for the formal run and supply their exact
commits. Every path below must be on shared storage and, when a container is
used, mounted at the same path inside the container:

```bash
export QWENAIR_BRIDGE_ROOT=<shared-source>/Megatron-Bridge
export QWENAIR_MCORE_ROOT=<shared-source>/Megatron-LM
export QWENAIR_TE_SOURCE_ROOT=<shared-source>/TransformerEngine
export QWENAIR_MODEL_CONFIG=<shared-source>/configs-and-numbers/sglang/agg/accuracy/numbers/provenance/bf16-model-config.json
export QWENAIR_RUN_DIR=<shared-scratch>/qwenair-target-1024
export QWENAIR_CACHE_ROOT=<shared-scratch>/qwenair-target-cache
export QWENAIR_PYTHON=<shared-venv>/bin/python

export QWENAIR_EXPECTED_BRIDGE_COMMIT=<Megatron-Bridge-commit>
export QWENAIR_EXPECTED_MCORE_COMMIT=<Megatron-LM-commit>
export QWENAIR_EXPECTED_TE_COMMIT=<Transformer-Engine-commit>
export QWENAIR_EXPECTED_MODEL_CONFIG_SHA256=b7d4f14b5891c998e767f92637e1f9c81eca57252f645b749b1a515e732aa816

# Optional Python overlay built against the container's Transformer Engine.
# Do not point this at the raw Transformer Engine source checkout.
# export QWENAIR_TE_PYTHON_OVERLAY=<shared-runtime>/te-python-overlay

# Optional fully cached/offline execution.
# export QWENAIR_HF_HUB_OFFLINE=1
# export QWENAIR_TRANSFORMERS_OFFLINE=1
# export QWENAIR_DATASETS_OFFLINE=1

# Optional Pyxis/Enroot container. Preserve every absolute path above: each
# host path must be mounted at the identical absolute path in the container.
# export QWENAIR_CONTAINER_IMAGE=<container-image.sqsh>
# export QWENAIR_CONTAINER_MOUNTS=<shared-storage>:<shared-storage>

mkdir -p "${QWENAIR_RUN_DIR}"
sbatch \
  --account=<B200-or-B300-account> \
  --partition=<B200-or-B300-partition> \
  --output="${QWENAIR_RUN_DIR}/sbatch-%j.out" \
  --export=ALL \
  "${QWENAIR_BRIDGE_ROOT}/examples/models/qwenair/slurm_target_multimodal.sh"
```

The launcher rejects a non-32-rank allocation, a commit mismatch, a dirty
source tree, or a config-hash mismatch before allocating model memory. It
writes a source manifest, the combined rank log, and TensorBoard events under
`QWENAIR_RUN_DIR`. Hugging Face, NeMo, XDG, Torch, Triton, TorchInductor, and
extension caches are redirected below `QWENAIR_CACHE_ROOT`; compiled caches
use one directory per rank to avoid concurrent writers. The same launcher
works on eight-GPU B200 and B300 nodes; select the hardware with the account
and partition supplied to `sbatch`. `QWENAIR_TE_SOURCE_ROOT` is used only to
verify the expected Git commit. If a compatible runtime overlay is supplied,
the launcher separately checks that its `qsa.py` is byte-identical to that
source commit before adding the overlay to `PYTHONPATH`.

Checkpoint writes and loads are disabled by default. Export
`QWENAIR_CHECKPOINT_DIR=<checkpoint-directory>` before submission only when
the run should resume or persist its final state; the CLI then saves at step
1024. `QWENAIR_TRAIN_ITERS` can be set to a smaller value for an explicit
bring-up gate, while the default remains the formal 1024-step run. The Qwen3.5-0.8B
processor revision is pinned because it matches the canonical 248,320-token
model embedding capacity and multimodal IDs. Before distributed setup, the CLI
loads that exact revision and verifies that its source config has a 248,320
text-vocabulary capacity, its tokenizer fits within that capacity, and all six
special tokens have their canonical IDs. At the pinned revision,
`tokenizer.vocab_size` is 248044 and `len(tokenizer)` is 248077 after added
tokens. The model EOS is `<|endoftext|>` (248044), while the processor's chat
EOS is `<|im_end|>` (248046). The
provenance directory does not contain a QwenAir processor or chat template, so
chat-template equivalence beyond that token contract remains an explicit
validation boundary.

The source JSON declares one MTP layer, but this recipe disables MTP because
the authoritative shift and loss objective is unavailable. At sequence length
128, the configured QSA indexer budget of 2048 is dense-equivalent; this avoids
depending on the unavailable separate indexer-training objective, but it does
not validate sparse selection. A successful loss curve therefore validates the
implemented text, vision, MoE, PLE, GDN, dense-equivalent QSA, and distributed
optimizer path within these stated boundaries.
