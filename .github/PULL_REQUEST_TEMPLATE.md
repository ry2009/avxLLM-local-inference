## What changed
- [ ] Code paths (files/dirs)
- [ ] Flags/env needed (if any)

## Why
- Link to acceptance gate(s) this PR targets.

## Validation
- [ ] `cmake --build build -j` & `ctest --output-on-failure` pass
- [ ] Perf run(s) executed locally:
  - [ ] tokenizers
  - [ ] gemm_lora
  - [ ] attn_prefill
  - [ ] attn_decode
  - [ ] sched_zipf
- Attach or paste key rows from `reports/*_summary.json`

## Gates
- [ ] Tokenizer ≥ 5× HF (or ≥3.5× on AVX2 lane)
- [ ] GEMM ≥ 1.4× at rank=32; LoRA tax ≤ 10%
- [ ] Prefill +≥25% ; Δpp ≤ 0.3
- [ ] Decode +≥15% ; Δpp ≤ 0.3 ; scratch ≤ 1.25×
- [ ] Scheduler tracking: p95 ≤ +15% ; LoRA tax ≤ 5% (target)

## Baselines
- [ ] No baseline changes
- [ ] Baselines updated (include `_meta` block)
