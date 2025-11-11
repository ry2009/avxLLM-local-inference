# INF-ENG Onboarding (CPU Inference Engine)

## Mission & Non-negotiables
- CPU-first runtime; PEFT (LoRA/DoRA) as first-class.
- Rowwise INT8 + FP16 scales (+ optional outliers) via **INFQ** format.
- Accuracy bars: Δperplexity ≤ 0.3pp vs FP16; LoRA tax ≤ 5% (P2) / ≤10% (P1).
- Perf bars: fused GEMM ≥1.4× baseline @ rank=32; prefill +≥25%; decode +≥15% (ctx≥2k).

## Repo Essentials
- Toolchain: **clang-17**, **cargo**, Python 3.11/3.12.
- Defaults: `INFENG_AVX512=OFF`, `INFENG_ENABLE_BENCHMARKS=ON`.
- Runners: standalone perf binaries under `tools/perf/cpp/*` (no network).

## Build & Test (mac or linux)
```bash
# mac: install llvm@17 + rust, set SDKROOT if needed
# linux: apt/yum clang-17 + cargo

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFENG_AVX2=ON -DINFENG_AVX512=OFF      # ON only if host supports it

cmake --build build -j
ctest --test-dir build --output-on-failure

# Python / NumPy sanity (used by converter tests)
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip "numpy==2.0.2" safetensors
python tools/py/check_numpy.py   # prints {"ok": true, "ver": "..."}
```

## Perf Sweeps (CSV + JSON, gates enforced in CI)
```bash
# Tokenizer
tools/perf/run_tokenizer_perf.sh

# GEMM (fused vs two-call; LoRA tax)
tools/perf/run_gemm_perf.sh

# Attention prefill (RoPE→FHT)
tools/perf/run_attn_prefill.sh

# Attention decode (INT8 QK, BF16 softmax/PV)
tools/perf/run_attn_decode.sh

# Scheduler Zipf (latency & LoRA tax under churn)
tools/perf/run_sched_zipf.sh
```

Artifacts land in `reports/*.csv` and `reports/*_summary.json`. Baselines live under `tools/perf/*_baseline.json`.

Refreshing baselines (maintainers only)
```bash
python tools/perf/compare_gemm.py            --csv reports/gemm_lora.csv            --update-baseline
python tools/perf/compare_attn_prefill.py    --csv reports/attn_prefill.csv         --update-baseline
python tools/perf/compare_attn_decode.py     --csv reports/attn_decode.csv          --update-baseline
python tools/perf/compare_sched_zipf.py      --csv reports/sched_zipf.csv           --update-baseline
```

Each baseline JSON includes `_meta` (cpu_model, cores, flags, sdk, timestamp) to make perf diffs explainable.

Acceptance Gates (CI will fail if violated)

- **Tokenizer:** ≥ 5× HF baseline @ 8 threads (temporary AVX2 lane: 3.5× min).
- **GEMM fused:** ≥ 1.4× vs two-call baseline at rank=32; LoRA tax ≤ 10% (P1).
- **Prefill FHT:** throughput +≥ 25% vs FP16 baseline; Δpp ≤ 0.3.
- **Decode:** throughput +≥ 15% vs FP16 at ctx≥2k; Δpp ≤ 0.3; scratch ≤ 1.25× baseline.
- **Scheduler (tracking in P1):** p95 ≤ base-only + 15%; LoRA tax ≤ 5% (target).

## Common CMake options
- `-DINFENG_AVX2=ON|OFF`
- `-DINFENG_AVX512=ON|OFF`
- `-DINFENG_ENABLE_BENCHMARKS=ON|OFF`

## Quick triage script
```bash
./tools/sys/diag.sh
# prints compiler versions, SDKROOT, CPU flags, numpy status, and AVX512 capability
```

## Minimal "Hello build" commands (fresh env cheat sheet)
```bash
# Build & unit tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DINFENG_AVX2=ON -DINFENG_AVX512=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure

# Perf (CSV + JSON)
tools/perf/run_tokenizer_perf.sh
tools/perf/run_gemm_perf.sh
tools/perf/run_attn_prefill.sh
tools/perf/run_attn_decode.sh
tools/perf/run_sched_zipf.sh

# Update a baseline (maintainers only, after verifying improvement)
python tools/perf/compare_attn_decode.py --csv reports/attn_decode.csv --update-baseline
```

## Quick Style Rules
- C++20, `.cc`/`.h`, clang-format LLVM style (2 spaces).
- Avoid exceptions/RTTI in hot paths.
- Runtime-visible headers live under `engine/include/infeng/...`.
- Kernels expose a plain C ABI shim for Python/bench bindings.
- CSV schemata documented at top of each bench; JSON summaries include `_meta` hardware block.
- Mirror any new flag in docs and CI matrices.

## Session Kick-off Snippet
Paste at the top of each new Codex session to restore context:
```
SYSTEM CONTEXT (pin this):
- Use clang-17 + cargo; default INFENG_AVX512=OFF, INFENG_ENABLE_BENCHMARKS=ON.
- Perf runs rely on standalone binaries in tools/perf/cpp/* (no Google Benchmark dependency).
- Perf gates & baselines: see docs/ONBOARDING.md (accuracy and throughput thresholds).
- INFQ = rowwise INT8 + FP16 scales (+ optional outliers with 64B-aligned coalesced blobs).
- Never regress gates without updating baselines and including a rationale in PR.
- If NumPy import fails, skip only converter tests; builds must still pass.
- For any new kernel, provide: (1) reference impl, (2) unit tests, (3) bench CSV + JSON summary, (4) CI compare script/gate update.
```

## Current Status Dashboard
- **Last completed:** Scheduler Zipf instrumentation landed; GEMM + prefill perf gates promoted to required; tokenizer perf gate reset to 5× with hardened build flags.
- **Open focus:** Finalise docs/diagnostics sweep and re-verify tokenizer/GEMM/prefill lanes on the Mac baseline.
- **Next planned:** Layer the local RL/SFT toolkit onto the stabilised engine before starting the BoN emergence predictor work.

Update this section as tasks progress so future sessions can resume quickly.
