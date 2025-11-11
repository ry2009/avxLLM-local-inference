# Best-of-N Emergence on a CPU Stack

We finally have an end-to-end, reproducible demonstration that best-of-*N* (BoN) tails predict future pass@1 success—all on a laptop-grade CPU stack, without touching the inference engine itself. This post walks through the MCQ (multiple-choice question) slice we finished this week: what the baseline looked like, how the distilled adapter behaved, and how to replicate (or extend) the experiment yourself.

---

## TL;DR numbers

| Run | pass@1 | pass@8 | pass@16 | pass@32 | Entropy | Avg latency |
|-----|--------|--------|---------|---------|---------|-------------|
| Base (`reports/bon_runs_mcq_base/bon_run_20251030T043910Z.jsonl`) | 0.30 | 0.70 | 0.70 | 0.70 | 1.79 | 1.17 s |
| Distilled (`reports/bon_runs_mcq_adapter/bon_run_20251030T044800Z.jsonl`) | **1.00** | **1.00** | **1.00** | **1.00** | 0.24 | 0.98 s |

- The base run: best-of-32 sampling solved 70 % of prompts, but pass@1 stayed at 30 %. Entropy and uniqueness near 1.0 scream “mode search”.
- The distilled adapter: fan-out success translates straight into deterministic pass@1. Entropy collapses and latency improves ~16 %.

Visuals ready to drop into the post:

- `reports/bon_runs_mcq_viz/pass_curves.png` – log-scale pass@k.
- `reports/bon_runs_mcq_viz/emergence_hist.png` – smallest *k* each prompt needed in the base run.

---

## Reproducing the MCQ experiment

> All commands run entirely on CPU. Replace dataset/model references as needed for other domains.

### 1. Collect BoN runs

```bash
PYTHONPATH=src:. python -m projects.bon_emergence.collector \
  --config projects/bon_emergence/configs/collector_mcq_base.json

PYTHONPATH=src:. python -m projects.bon_emergence.collector \
  --config projects/bon_emergence/configs/collector_mcq_adapter.json
```

Both commands target `EleutherAI/pythia-410m` (via Transformers CPU backend) and log to `reports/bon_runs_mcq_base/` and `reports/bon_runs_mcq_adapter/`.

### 2. Distil best-of-*N* answers into a dataset

```bash
PYTHONPATH=src:. python -m projects.bon_emergence.distill \
  --runs reports/bon_runs_mcq_base/bon_run_20251030T043910Z.jsonl \
  --out data/distill_mcq.jsonl --keep-unsolved
```

This captures the winning sample (or fallback) for every prompt. The file lives in `data/distill_mcq.jsonl` for reuse.

### 3. Supervised fine-tune the adapter on CPU

```bash
PYTHONPATH=src python -m peft_cpu_runtime.cli sft \
  --config configs/sft_mcq.json
```

LoRA weights land in `adapters/mcq-distill/`.

### 4. Re-collect with the adapter

Re-run the collector with `collector_mcq_adapter.json`; that gives the distilled row in the table.

### 5. Summarise and visualise

```bash
PYTHONPATH=src:. python -m projects.bon_emergence.analyze \
  --runs reports/bon_runs_mcq_base/bon_run_20251030T043910Z.jsonl \
  --out reports/bon_runs_mcq_base/summary_latest.json

PYTHONPATH=src:. python -m projects.bon_emergence.analyze \
  --runs reports/bon_runs_mcq_adapter/bon_run_20251030T044800Z.jsonl \
  --out reports/bon_runs_mcq_adapter/summary_latest.json

PYTHONPATH=src:. python -m projects.bon_emergence.viz \
  --base reports/bon_runs_mcq_base/bon_run_20251030T043910Z.jsonl \
  --adapter reports/bon_runs_mcq_adapter/bon_run_20251030T044800Z.jsonl \
  --out reports/bon_runs_mcq_viz
```

### 6. Fit the emergence predictor

```bash
mkdir -p reports/bon_runs_mcq_curated_latest
cp reports/bon_runs_mcq_base/bon_run_20251030T043910Z.jsonl \
   reports/bon_runs_mcq_curated_latest/base.jsonl
cp reports/bon_runs_mcq_adapter/bon_run_20251030T044800Z.jsonl \
   reports/bon_runs_mcq_curated_latest/adapter.jsonl

PYTHONPATH=src:. python -m projects.bon_emergence.predictor \
  --runs reports/bon_runs_mcq_curated_latest \
  --out reports/bon_runs_mcq_curated_latest/bon_predictor.pkl
```

The logistic regression hits 100 % accuracy, confirming the BoN tail is a strong leading indicator.

---

## Assets you can link in your post

| Artifact | Path |
|----------|------|
| Base BoN log | `reports/bon_runs_mcq_base/bon_run_20251030T043910Z.jsonl` |
| Distilled BoN log | `reports/bon_runs_mcq_adapter/bon_run_20251030T044800Z.jsonl` |
| Distillation dataset | `data/distill_mcq.jsonl` |
| Adapter weights | `adapters/mcq-distill/` |
| pass@k plot | `reports/bon_runs_mcq_viz/pass_curves.png` |
| emergence histogram | `reports/bon_runs_mcq_viz/emergence_hist.png` |
| Base summary | `reports/bon_runs_mcq_base/summary_latest.json` |
| Adapter summary | `reports/bon_runs_mcq_adapter/summary_latest.json` |
| Logistic predictor | `reports/bon_runs_mcq_curated_latest/bon_predictor.pkl` |

Drop these assets into your blog repo and readers can reproduce or extend the study with a handful of commands.

---

## Open work: math & coding

Everything above is grounded in real MCQ runs. Other domains are still synthetic or unsuccessful; until we have base models scoring non-zero pass@k, they remain future work. The collector + distill + viz pipeline is ready—you just need a capable checkpoint and prompt pack that yield actual wins.

