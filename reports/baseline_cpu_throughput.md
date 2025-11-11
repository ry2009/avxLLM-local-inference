# Baseline CPU LoRA Serving Prototype

## Environment
- Host: Intel(R) Core(TM) i9-9880H @ 2.30 GHz (8C/16T), 16 GB RAM.
- Python 3.12 virtual environment (`python3 -m venv .venv`).
- Dependencies: see `requirements.txt` (notably `torch==2.2.2`, `peft==0.17.1`, `transformers==4.45.2`, `numpy==1.26.4`).

## Reproduction Steps
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter therapy=usmanalam82/tinyllama-therapy-lora \
  --adapter story=dasrupdip04/lora-finetuned-TinyLLama \
  --max-new-tokens 16 \
  --warmup 0 \
  --iters 1
```

## Observed Metrics (single batch of 3 prompts)
| Metric | Value |
| --- | --- |
| Avg latency per iteration | 4.82 s |
| Sequences per second | 0.62 |
| Tokens per second | 3.32 |
| Total generated tokens | 16 |

## Notes
- The runtime hot-swaps LoRA adapters via `PeftModel.set_adapter` without reloading the base model.
- Deterministic decoding (`temperature=0`, `top_p=0.95`, `do_sample=False`) keeps results comparable across runs; enable `--do-sample` for qualitative inspection.
- Throughput is currently constrained by single-threaded Python generation; profiling hooks will be added before attempting kernel-level optimizations.
- Updated benchmark harness now supports `--zipf-alpha` and `--profile` flags for Zipf-distributed adapter sampling and tokenizer vs decode timing breakdowns (see `docs/benchmark_usage.md`).
- Quantized `.gguf` models can be evaluated via `--engine llama_cpp`, enabling apples-to-apples comparisons against llama.cpp baselines while reusing the same Zipf trace inputs.

## Next Actions
- Capture multi-batch Zipfian adapter distributions to stress-test adapter switching.
- Instrument CPU profilers (e.g., `torch.profiler`, `py-spy`) to identify batching and tokenizer bottlenecks.
- Integrate quantized base models (GGUF or `bitsandbytes` 4-bit) to reduce memory pressure ahead of GPU experiments.
