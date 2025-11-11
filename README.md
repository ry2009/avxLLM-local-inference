# avxLLM-local-inference
![Build](https://github.com/ry2009/avxLLM-local-inference/actions/workflows/build.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
> CPU-native inference stack that fuses C++/Rust kernels, quantized weights, and hot-swappable LoRA/RL adapters to deliver GPU-class throughput on commodity AVX/AMX hardware.

---
## TL;DR
- **Real speedups:** AVX int8 decode/prefill kernels run up to 2.5× faster than fp16 baselines and the Rust tokenizer pushes ~28K tokens/s per core (see `reports/perf_report.md` and the Plotly dashboard).
- **Full CPU AI workflow:** Pre-training, SFT, RL, hybrid adapter blending, and evaluation scripts all run locally—no GPU required.
- **Observability-first:** Every helper emits TPS/TTFT JSON/CSV so regressions fail CI (`python-smoke` job) and perf dashboards stay current.
- **Drop-in runtime knobs:** `INFENG_NUM_THREADS`, `INFENG_NUM_INTEROP_THREADS`, and `INFENG_TOKEN_CACHE` let you pin PyTorch pools and bypass repeated tokenization.

---
## Quick Links
- Architecture & roadmap: `docs/engine_vision.md`, `docs/engine_roadmap.md`
- Performance dashboards: `scripts/generate_perf_dashboard.py` → `reports/perf_dashboard.html`
- CPU AI playbooks: `docs/prepost_playbook.md`, `docs/throughput_playbook.md`, `docs/rust_speedups.md`
- Lab checklist (demo order + artifacts): `docs/lab_review_checklist.md`

---
## Getting Started
### Python Runtime (LoRA / RL experiments)
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
Run inference:
```bash
peft-cpu-run infer meta-llama/Llama-3.2-1B \\
  "Write a haiku about AVX kernels" \\
  --adapter instruct=/path/to/lora \\
  --max-new-tokens 64 --telemetry
```
Set `INFENG_NUM_THREADS=<cores>` / `INFENG_NUM_INTEROP_THREADS=<cores>` and optionally `INFENG_TOKEN_CACHE=<size>` before running scripts for deterministic throughput.

### Native C++ Engine
```bash
cmake -S . -B build -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DINFENG_AVX2=ON -DINFENG_AVX512=ON -DINFENG_ENABLE_PYTHON=ON
cmake --build build -j$(sysctl -n hw.logicalcpu)
ctest --test-dir build --output-on-failure
```
Requires CMake ≥3.25, clang/LLVM with AVX2, Ninja/Make, Python 3.12, and Rust (`cargo`) for tokenizer codegen.

---
## Workflows at a Glance
| Goal | Command / Script | Output |
| --- | --- | --- |
| Warm up models/adapters | `python tools/download_assets.py ...` or `make run-manifest` | Downloaded HF weights into `models/` / `adapters/` |
| Quick inference demo | `scripts/run_local_inference.py` | JSON outputs + telemetry CSV |
| CPU LoRA SFT | `scripts/run_local_training.py --limit 64` | `adapters/quickstart-trained/` |
| CPU pre-training | `scripts/run_local_pretrain.py` | `checkpoints/tiny-pretrain-cpu/` |
| CPU RL fine-tune | `scripts/run_rl_demo.py --config configs/rl_tiny.json` | `adapters/rl-tiny/` |
| Reward delta (pre vs post) | `scripts/run_rl_eval.py --model-id ... --adapter adapters/rl-tiny` | `reports/rl_eval.json` |
| Prompt-level telemetry | `scripts/run_prompt_benchmark.py --model-id ... --prompts data/math_prompts.jsonl` | CSV+JSON with TPS/TTFT per prompt |
| Adapter blend | `scripts/blend_lora_adapters.py --adapter-a A --adapter-b B --alpha 0.6 --output blend` | Hybrid adapter directory |
| llama.cpp comparison | `scripts/run_llama_compare.py --model-id ... --llama-model path.gguf` | `reports/llama_compare.json` |
| Throughput sweep (regression gate) | `scripts/run_throughput_sweep.py --lengths 32,64,128 --min-tps ...` | `reports/throughput_sweep.json` |
| Telemetry matrix | `scripts/run_telemetry_matrix.py --adapter demo=...` | `reports/telemetry_matrix.json` |
| Perf dashboard | `scripts/generate_perf_dashboard.py` | `reports/perf_dashboard.html` |
| Rust tokenizer summary | `scripts/summarize_tokenizer_perf.py` | Prints best threads/tokens-s |

All scripts are unit-tested (see `src/peft_cpu_runtime/tests/test_scripts.py`).

---
## Observability & Performance
- **Telemetry everywhere:** `CpuPeftRuntime.enable_profiling()` streams TPS/TTFT per adapter. CLI helpers expose `--telemetry` flags; CI uploads `reports/ci_smoke_metrics.json` on every push.
- **Plotly dashboards:** `scripts/generate_perf_dashboard.py` combines `reports/attn_*` + `reports/tokenizer.csv` into an HTML summary.
- **Rust tokenizer insights:** `docs/rust_speedups.md` explains the Rust/C ABI bridge and how to rerun `cargo bench`; `scripts/summarize_tokenizer_perf.py` highlights the optimal thread count from `reports/tokenizer.csv`.
- **Comparative baselines:** `scripts/run_llama_compare.py` and `scripts/run_throughput_sweep.py --min-tps ...` provide hard numbers against llama.cpp/older kernels. CI fails if TPS or TTFT regresses.

---
## AI Playbooks (CPU-only)
- **Pre-training:** `scripts/run_local_pretrain.py` + `docs/prepost_playbook.md`
- **SFT (LoRA):** `scripts/run_local_training.py`, eval via `scripts/run_local_eval.py`
- **RL:** `scripts/run_rl_demo.py` (PPO-style) + `scripts/run_rl_eval.py` for reward deltas
- **Hybrid adapters:** `scripts/blend_lora_adapters.py` mixes any two adapters without retraining
- **Prompt-level QA:** `scripts/run_prompt_benchmark.py` gives per-prompt TPS/TTFT, great for lab deep dives

---
## Docs & References
- Architecture: `docs/engine_vision.md`
- Roadmap: `docs/engine_roadmap.md`
- Perf/telemetry: `docs/perf_metrics.md`, `reports/perf_report.md`, `docs/rust_speedups.md`
- CPU playbooks: `docs/prepost_playbook.md`, `docs/throughput_playbook.md`
- Lab checklist & artifacts: `docs/lab_review_checklist.md`

---
## Make Targets
`make setup` · `make cpp-build` · `make test` · `make py-test`
`make run-infer` · `make run-train` · `make run-pretrain`
`make run-eval` · `make run-pipeline` · `make run-telemetry`
`make run-throughput` · `make run-rl-demo` · `make run-rl-eval`
`make run-perf-dashboard` · `make run-manifest` · `make run-ci-smoke` · `make check-mac`

---
## Repo Tour
| Path | Description |
| --- | --- |
| `engine/` | AVX/AMX kernels, tokenizer bridge (Rust crate), telemetry hooks, memory manager |
| `src/peft_cpu_runtime/` | Python runtime + CLI (SFT, RL, converter utilities) |
| `configs/` | Ready-to-run JSON configs (SFT, RL, throughput sweeps) |
| `scripts/` | Plug-and-play helpers listed above |
| `tools/` | C++ perf harnesses (`bench`, `perf`, `eval`) |
| `docs/` | Design notes, playbooks, onboarding |
| `reports/` | Artifacts emitted by scripts / CI (`*.csv`, `*.json`, dashboards) |

---
## License
MIT — see `LICENSE`.
