# End-to-End Pipeline Quickstart

Use `scripts/run_end_to_end.py` when you want to demonstrate the full CPU LoRA
workflow—downloading a base model, fine-tuning on a tiny dataset, and running a
prompt suite with telemetry—on a single macOS laptop.

```bash
python scripts/run_end_to_end.py \
  --model-id sshleifer/tiny-gpt2 \
  --dataset data/distill_pairs.jsonl \
  --train-samples 32 \
  --eval-prompts data/math_prompts.jsonl \
  --telemetry
```

## What it does
1. Downloads the base model into `models/pipeline-base/` (respects `HF_TOKEN`).
2. Builds a `TrainingConfig` that points at `data/distill_pairs.jsonl` and
   fine-tunes a LoRA adapter under `adapters/pipeline/`.
3. Runs the evaluation prompts via `CpuPeftRuntime`, prints JSON responses, and
   optionally emits TPS/TTFT metrics if `--telemetry` is set.

## Options
- `--reuse-adapter` lets you skip training and reuse a previously exported LoRA
  directory.
- `--dataset-field` controls which key to read from JSON/JSONL files (defaults
  to `text`).
- `--eval-limit` trims the number of prompts when you only need a quick smoke
  test.

Pair this script with `scripts/check_mac_env.py` (or `make check-mac`) before a
live demo so you know the toolchain is ready.
