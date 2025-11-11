# avxLLM-local-inference
> CPU-native LLM inference engine that fuses base weights, quantized formats, and hot-swappable LoRA adapters to deliver GPU-class throughput on commodity AVX/AMX hardware.

## Why This Exists
- **CPU-first LoRA serving:** Reuses a single base model while hot-swapping multiple adapters with near-zero overhead.
- **Fused, quantized kernels:** Custom C++/SIMD kernels (AVX2/AVX-512/AMX) pair base weights with LoRA deltas and int4/int8/FP8 formats to keep bandwidth low.
- **Composable runtime layers:** Scheduler, memory manager, tokenizer, and telemetry live in distinct modules so that each can evolve independently.
- **Visibility baked in:** Every stage (tokenizer → scheduler → kernel) emits metrics for perf dashboards and regression gates.

## Repository Layout
| Path | What lives here |
| --- | --- |
| `engine/` | C++20 core (kernels, scheduler, kv cache, quantization, tokenizer bridge, telemetry hooks). Built via CMake and ships a C ABI for bindings. |
| `src/peft_cpu_runtime/` | Python reference runtime + CLI. Mirrors the native scheduler patterns and powers quick LoRA experiments. |
| `apis/python/` | pybind11 bindings that expose the native engine to Python clients. |
| `tools/bench`, `tools/perf`, `tools/eval` | Microbenchmarks, perf automation, and evaluation harnesses used in CI/regression runs. |
| `docs/` | Architecture vision, roadmap, quantization notes, blog drafts. Start with `engine_vision.md` and `engine_roadmap.md`. |
| `configs/` | Example JSON/YAML configs for training, inference traces, schedulers, and telemetry. |
| `tests/` | C++ unit/integration tests (fused GEMM, attention, scheduler Zipf traces, tokenizer goldens). Executed via `ctest`. |
| `reports/` | CSV/JSON outputs from benchmarks and telemetry exporters. |

## Quick Start
### 1. Python LoRA Runtime (fastest path)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
Run inference with a base HF model and one or more adapters:
```bash
peft-cpu-run infer meta-llama/Llama-3.2-1B \
  "Write a haiku about AVX kernels" \
  --adapter instruct=/path/to/lora \
  --max-new-tokens 64 --temperature 0.2 --telemetry
```
Outputs are printed as JSON. Enable telemetry (`--telemetry`) to stream runtime metrics into `reports/runtime_metrics.csv`.

Need sample weights fast? Use the downloader (set `HF_TOKEN` if the repos are gated):
```bash
python tools/download_assets.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model-dir models/tinyllama-chat \
  --adapter-id theone049/agriqa-tinyllama-lora-adapter --adapter-dir adapters/agriqa
```
Prefer a single command? `scripts/run_local_inference.py` will download `sshleifer/tiny-gpt2`, load an optional adapter, and emit telemetry in one go.

### 2. Native Engine Build (C++ kernels)
```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFENG_AVX2=ON -DINFENG_AVX512=ON -DINFENG_ENABLE_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```
Requirements: CMake ≥3.25, Ninja/Make, Python 3.12, a C++20 compiler with AVX2 support, and Rust `cargo` (for tokenizer codegen). Generated headers land in `build/generated/`.

### 3. Telemetry & Benchmarks
- `tools/bench` — Google Benchmark harness for GEMMs, attention, tokenizer throughput.
- `tools/perf` — wrappers for `perf stat`, VTune, and CSV exports consumed by the `reports/` dashboards.
- `reports/benchmark_usage.md` shows how nightly perf gates are configured.

### 4. Plug-and-Play Scripts
- `scripts/run_local_inference.py` — downloads `sshleifer/tiny-gpt2` (or the model you choose), runs prompts via `CpuPeftRuntime`, and prints TPS/TTFT metrics.
- `scripts/run_local_training.py` — fine-tunes a tiny LoRA adapter on `data/distill_math.jsonl`; outputs land in `adapters/quickstart-trained`.

## Runtime Features
- Batched inference via `RequestBatch` objects with shared `InferenceTraceConfig` (temperature, top-p, stop sequences, etc.).
- Adapter-aware scheduling that groups prompts per adapter, activates LoRA weights on demand, and overlaps tokenization with generation using thread pools.
- Hookable telemetry: enable via CLI flag or programmatically to flush CSV metrics and feed dashboards.
- Training utilities: supervised fine-tuning (`sft`), lightweight RL (`rl`), and causal pre-training (`pretrain`) live under `src/peft_cpu_runtime/training` with shared dataset + reward helpers.

## Architecture Preview
- **Scheduler** assembles microbatches, respects Zipfian adapter distributions, and partitions cores across decode vs. adapter streams.
- **Memory Manager** owns KV cache pages, adapter weights, and quantized tensors with NUMA-aware placement.
- **Kernel backend** implements fused base+LoRA matmuls, hybrid attention (FHT + FP8/FP16 mixes), and normalization/activation paths.
- **Quantization pipeline** converts Hugging Face `safetensors` into engine-native packed layouts (block int4/int8 with per-row scales) and manages calibration traces.
- **Tokenizer service** keeps the Rust-backed tokenizer hot, caches prefixes, and streams tokens into the decode loop to hide CPU overhead.
Read `docs/engine_vision.md` and `docs/engine_roadmap.md` for the full breakdown of milestones, risks, and mitigation plans.

For a telemetry deep dive (TPS, TTPS, TTFT, Rust↔C++ hand-off), see `docs/perf_metrics.md`.
For macOS-specific toolchain notes (Homebrew, LLVM, Python/Rust steps), see `docs/mac_setup.md`.

## Development Workflow
1. Prototype in Python (`peft_cpu_runtime`) to validate scheduler or telemetry ideas.
2. Port hot paths into `engine/` kernels, backing them with Catch2/GoogleTest fixtures in `tests/`.
3. Run `tools/bench` microbenchmarks and capture results under `reports/`.
4. Keep docs fresh—new design decisions go into `docs/*.md` alongside code reviews.

## Contributing
- Follow the onboarding notes in `docs/ONBOARDING.md` for environment setup and coding standards.
- Add or update docs whenever you touch architecture-critical code (scheduler, quantization, telemetry, tokenizer).
- Include perf numbers and regression tests with every kernel change; telemetry is mandatory for any new long-running feature.

## License
Released under the MIT License (see `LICENSE`).

## CI & Secrets
`docs/ci_setup.md` lists the optional `HF_TOKEN` secret so GitHub Actions can prefetch
Hugging Face artifacts without hitting anonymous rate limits.
