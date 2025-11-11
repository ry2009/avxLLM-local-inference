# BoN Emergence Predictor (Local CPU Prototype)

This folder hosts a **separate project** that *uses* the inference engine and the
training toolkit, but does not extend them. The goal is to predict when
“best-of-N” (BoN) sampling produces a qualitative jump in accuracy, entirely on
the Mac CPU stack that we’ve stabilised.

## What’s here

| File / Dir | Purpose |
|------------|---------|
| `collector.py` | Runs batched sampling against the runtime, scores completions, records pass@k, diversity, and latency stats. |
| `distill.py` | Builds a best-of-N imitation dataset from collector output for SFT. |
| `analyze.py` | Summarises runs (mean pass@k, entropy, cost) to compare adapters. |
| `predictor.py` | Fits a logistic regression on the collected features to predict BoN emergence. |
| `configs/` | Example JSON configs for SFT/RL/pre-train runs and for the BoN collector. |
| `prompts.jsonl` | Minimal prompt set with deterministic answers. |

## Quick start

1. **Fine-tune or RL-train an adapter** (optional)

   Re-use the CLI introduced earlier, e.g.

   ```bash
   peft-cpu-run sft --config configs/sft_example.json

   peft-cpu-run rl \
     --config configs/rl_example.json \
     --reward builtin:exact_match
   ```

   The adapters land under `adapters/…`.

2. **Collect BoN runs**

   ```bash
   python projects/bon_emergence/collector.py \
     --config projects/bon_emergence/configs/collector_base.json
   ```

   This generates `reports/bon_runs/<timestamp>.jsonl` with per-sample scores,
   latency, pass@k curves, and diversity metrics.

3. **Distil “best-of-N” outputs (optional)**

   ```bash
   python projects/bon_emergence/distill.py \
     --runs reports/bon_runs/bon_run_*.jsonl \
     --out data/distill_pairs.jsonl --keep-unsolved

  peft-cpu-run sft \
    --config configs/sft_example.json
  ```

   Rerun the collector with the new adapter to compare curves.

   ```bash
   python projects/bon_emergence/collector.py \
     --config projects/bon_emergence/configs/collector_adapter.json
   ```

4. **Summarise & fit a predictor**

   ```bash
   python projects/bon_emergence/predictor.py \
     --runs reports/bon_runs \
     --out reports/bon_runs/bon_predictor.pkl
   ```

   The script expects `scikit-learn`; install it in your virtualenv with
   `pip install scikit-learn` if it’s missing.

   For quick comparisons without fitting a model:

   ```bash
   python projects/bon_emergence/analyze.py \
     --runs reports/bon_runs \
     --out reports/bon_runs/summary.json
   ```

## Config schema (collector)

```jsonc
{
  "base_model": "EleutherAI/pythia-70m-deduped",
  "adapter_name": "demo-distill",
  "adapter_path": "adapters/demo-distill/demo-distill",
  "prompts_file": "projects/bon_emergence/prompts.jsonl",
  "answer_field": "answer",
  "k_values": [4, 8, 16],
  "samples_per_prompt": 8,
  "max_new_tokens": 12,
  "temperature": 0.1,
  "top_p": 0.9,
  "scorer": "exact",
  "output_dir": "reports/bon_runs",
  "metadata": {
    "experiment": "rl-demo",
    "notes": "first local sweep"
  }
}
```

## Notes

- Telemetry is optional. If you enable it (`--telemetry` when using the CLI) the
  runtime will write queue-wait CSVs that the collector can ingest as additional
  features.
- Everything here stays **outside** the core engine/toolkit so we can iterate on
  research ideas without complicating the base libraries.
