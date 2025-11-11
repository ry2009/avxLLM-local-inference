## GEMM Benchmark Usage

Build the benchmark target (requires `INFENG_ENABLE_BENCHMARKS=ON` during the CMake configure step):

```bash
cmake --build build --target bench_gemm_lora
```

Run the benchmark (single iteration per configuration) and record metrics:

```bash
./build/tools/bench/bench_gemm_lora --benchmark_display_aggregates_only=true
```

Results append to `reports/gemm_lora.csv` with fused vs baseline latency, LoRA tax, and throughput figures.
# Benchmark Harness Usage

## Zipfian Multi-Adapter Traces
Generate a synthetic batch of requests that follows a Zipf distribution over adapters plus the base model:
```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter therapy=usmanalam82/tinyllama-therapy-lora \
  --adapter story=dasrupdip04/lora-finetuned-TinyLLama \
  --num-requests 32 \
  --zipf-alpha 1.3 \
  --seed 123 \
  --include-base \
  --max-new-tokens 32 \
  --warmup 1 \
  --iters 3 \
  --metrics-out reports/tinyllama_zipf.json
```
- `--zipf-alpha` controls skew (larger = more weight on the first adapter). Set `<=0` to fall back to round-robin sampling.
- `--include-base` adds the base model (no adapter) to the sampling pool.
- `--metrics-out` persists aggregated + per-iteration metrics for later analysis.

## Profiling Tokenizer vs Decode Time
Enable lightweight profiling to capture per-adapter tokenization/generation timings:
```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter story=dasrupdip04/lora-finetuned-TinyLLama \
  --num-requests 8 \
  --max-new-tokens 16 \
  --profile
```
Each iteration now reports a `profiling` block with the latency spent tokenizing inputs and performing `generate()` calls per adapter group.

## Quantized GGUF (llama.cpp) Runtime
To benchmark a quantized `.gguf` model with llama.cpp bindings:
```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --engine llama_cpp \
  --base-model ./models/tinyllama-q4.gguf \
  --adapter legal=./adapters/legal-lora.gguf \
  --num-requests 16 \
  --zipf-alpha 1.1 \
  --seed 7 \
  --max-new-tokens 48 \
  --n-ctx 4096 \
  --n-threads 8
```
- LoRA adapters must be stored in llama.cpp-compatible format (`.gguf`). The runtime loads all adapters upfront and toggles them per request using llama.cpp’s native adapter APIs.
- Note: the prebuilt `llama-cpp-python` wheel (v0.2.90) does not expose LoRA adapter symbols (`llama_adapter_lora_init`). Build llama.cpp from source with LoRA support if adapter initialization fails.
- `--lora-scale` adjusts adapter scaling; leave at `1.0` unless the adapter author specifies otherwise.
- Profiling is not yet available for the llama.cpp path; the `--profile` flag is ignored.

## Comparing Against External Runtimes
1. **llama.cpp (CPU):** run the server with matching prompts/adapters (GGUF LoRA hot-swap) and collect tokens/sec for the same Zipf trace.
2. **kTransformers + HybriMoE:** leverage their hybrid CPU scheduling to evaluate whether adapter batching or stream partitioning beats our runtime on identical traces.
3. Normalize results by prompt mix and rank to target the ≥10% throughput improvement milestone described in `docs/benchmark_roadmap.md`.
