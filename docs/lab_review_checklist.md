# Lab Review Checklist

Use this script bundle when walking a top AI lab through the CPU inference
engine. Everything below runs on a single macOS workstation.

1. **Environment sanity**
   - `make check-mac` → verifies `cmake`, `ninja`, `clang/llvm`, `cargo`, `python`.
   - `make run-manifest` → downloads the base TinyGPT + AgriQA adapter (see
     `configs/sample_assets.json`).

2. **Telemetry sanity**
   - `make run-ci-smoke` → downloads the base model (if needed), generates a
     single prompt, and writes `reports/ci_smoke_metrics.json` (TPS + TTFT).

3. **Full pipeline**
   - `make run-pipeline` → executes `scripts/run_end_to_end.py` to train a tiny
     LoRA adapter on `data/distill_pairs.jsonl` and immediately evaluates
     `data/math_prompts.jsonl` with telemetry.

4. **Adapter matrix**
   - `make run-telemetry` (or run `scripts/run_telemetry_matrix.py` manually)
     → iterates over multiple adapters, generating `reports/telemetry_matrix.json`.
   - `make run-throughput` → sweeps prompt lengths and records
      `reports/throughput_sweep.json`, failing if TPS regresses.
   - `python scripts/run_prompt_benchmark.py --model-id ... --prompts data/math_prompts.jsonl`
     → produces per-prompt CSV/JSON for labs that want raw numbers.

5. **Artifacts to share with reviewers**
   - `reports/ci_smoke_metrics.json`
   - `reports/telemetry_matrix.json`
   - `reports/local_eval_metrics.json`
   - `reports/perf_report.md`
   - RL adapter from `scripts/run_rl_demo.py` (shows post-training manipulations).

6. **Optional**
   - `make run-eval` for deterministic JSON prompt files.
   - `make run-pretrain` to produce a local CPU checkpoint in
     `checkpoints/tiny-pretrain-cpu`.

Pair this checklist with the architecture vision (`docs/engine_vision.md`) and
perf/TTFT notes (`docs/perf_metrics.md`) when presenting to labs.
