# Pre/Post Training Wins

## Pre-training (CPU-only)
1. `make run-manifest` – warm up cache with TinyGPT.
2. `python scripts/run_local_pretrain.py --epochs 1 --batch-size 2` – produces
   `checkpoints/tiny-pretrain-cpu` (≈8 MB) showing the pre-train pipeline works
   entirely on CPU.
3. `python scripts/run_local_inference.py --model-id checkpoints/tiny-pretrain-cpu --prompts "Explain AVX."`

## Post-training / LoRA fine-tune
1. `python scripts/run_local_training.py --limit 64 --epochs 1` – writes
   `adapters/quickstart-trained`.
2. `python scripts/run_local_eval.py --adapter-id adapters/quickstart-trained --telemetry` – compare
   outputs vs base prompts; telemetry CSV proves the adapter swap overhead.

## Hybrid adapter blend
1. Train/obtain two adapters (`adapterA`, `adapterB`).
2. `python scripts/blend_lora_adapters.py --adapter-a adapters/A --adapter-b adapters/B --alpha 0.6 --output adapters/blend`.
3. Use `--adapter blend=adapters/blend` with `scripts/run_throughput_sweep.py` to
   evaluate the blended adapter.

## Reporting wins
- Use `scripts/run_ci_smoke.py` + `scripts/run_throughput_sweep.py --min-tps ...`
  to capture before/after metrics whenever you touch kernels or adapters.
- Store `reports/ci_smoke_metrics.json`, `reports/throughput_sweep.json`, and
  `reports/perf_report.md` as attachments when syncing with labs.
