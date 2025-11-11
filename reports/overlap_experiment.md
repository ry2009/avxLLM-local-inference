# Tokenizer Overlap Prototype (October 22, 2025)

Commands:
```bash
# Baseline (no overlap)
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --engine torch \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter therapy=usmanalam82/tinyllama-therapy-lora \
  --adapter story=dasrupdip04/lora-finetuned-TinyLLAma \
  --num-requests 6 \
  --zipf-alpha 1.3 \
  --seed 123 \
  --max-new-tokens 16 \
  --warmup 0 \
  --iters 1 \
  --include-base \
  --profile \
  --metrics-out /tmp/no_overlap.json

# Threaded tokenizer overlap (2 workers)
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --engine torch \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter therapy=usmanalam82/tinyllama-therapy-lora \
  --adapter story=dasrupdip04/lora-finetuned-TinyLLAma \
  --num-requests 6 \
  --zipf-alpha 1.3 \
  --seed 123 \
  --max-new-tokens 16 \
  --warmup 0 \
  --iters 1 \
  --include-base \
  --profile \
  --tokenize-overlap-workers 2 \
  --metrics-out /tmp/overlap.json
```

## Results
| Setting | Tokens/sec | Seq/sec | Avg Latency (s) | Therapy Tokenize (s) | Base Tokenize (s) |
| --- | --- | --- | --- | --- | --- |
| No overlap | 3.43 | 0.49 | 12.26 | 0.0014 | 0.00033 |
| Overlap (2 workers) | 3.51 | 0.50 | 11.95 | 0.0046 | 0.0041 |

## Observations
- Threaded tokenizer overlap delivered a modest +2.5% throughput improvement (3.43 → 3.51 tokens/sec) by precomputing tokenization while the previous adapter group decoded.
- Tokenization time per group increased as work moved onto worker threads, but GPU/CPU decode time fell slightly (11.94s → 11.64s) for the dominant adapter group.
- Further gains will require deeper overlap (e.g., concurrent adapter GEMMs) and reducing the Python GIL contention observed in the tokenization workers.
