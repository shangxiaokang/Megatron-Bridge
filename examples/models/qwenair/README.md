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
  --output-dir /shared/logs/qwenair-real-100-analysis
```

The analyzer requires all 100 steps, finite metrics, and zero skipped/NaN iterations. It compares steps 11–20 with 91–100 and fits a post-warmup linear trend. A `PASS` requires at least a 2% mean loss reduction, a negative slope, and at least seven of the final ten points below the early-window median.
