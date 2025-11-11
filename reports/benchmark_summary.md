# Benchmark Matrix Snapshot (October 22, 2025)

Generated via:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_benchmark_matrix.py \
  --config configs/benchmark_sample.json \
  --output reports/benchmark_matrix.json
```

## Tokens/sec Comparison
| Run | Engine | Base Model | Tokens/sec | Sequences/sec | Avg Latency (s) |
| --- | --- | --- | --- | --- | --- |
| torch_tiny_gpt2 | torch | sshleifer/tiny-gpt2 | 31.42 | 3.93 | 1.02 |
| torch_tinyllama_lora | torch | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0.98 | 0.49 | 8.16 |

## Notes
- Metrics pulled from `reports/benchmark_matrix.json` (single iteration, deterministic decoding). Tokens/sec chart saved to `reports/benchmark_tokens.html` with CSV export `reports/benchmark_tokens.csv`.
- Llama.cpp run with TinyLlama GGUF adapter failed because the pip wheel lacks LoRA adapter symbols (`llama_adapter_lora_init`); rebuilding llama.cpp with LoRA support is required before GGUF adapters can be benchmarked.
- Replace the sample config with additional workloads as adapters/GGUF conversions become available to broaden the comparison matrix.
- Warning messages in `stderr` highlight that `temperature=0` with `top_p=0.95`; adjust configs if sampling is desired.
