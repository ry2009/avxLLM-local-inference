# BoN Emergence (MCQ slice)

## MCQ slice

- Base run (`reports/bon_runs_mcq_base/bon_run_20251030T043910Z.jsonl`)
  - pass@1 = 0.30, pass@8/16/32 ≈ 0.7.
  - entropy 1.79, unique_frac 1.0, avg latency 1.17 s.
- Distilled adapter (`reports/bon_runs_mcq_adapter/bon_run_20251030T044800Z.jsonl`)
  - pass@1 = 1.00, pass@k >= 1 for k ≥ 4.
  - entropy 0.24, unique_frac 0.25, avg latency 0.98 s.
- Visuals ready at `reports/bon_runs_mcq_viz/pass_curves.png` and `reports/bon_runs_mcq_viz/emergence_hist.png`.
- Distillation dataset `data/distill_mcq.jsonl`; LoRA weights `adapters/mcq-distill/`.
- Logistic predictor (`reports/bon_runs_mcq_curated_latest/bon_predictor.pkl`) fitted on real runs (accuracy 1.0).
