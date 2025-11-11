# CPU-First PEFT Benchmarking & Optimization Roadmap

## Benchmark Coverage
- **Workload suites:** 
  - Synthetic Zipf-distributed adapter load traces with concurrent requests (1–16 adapters, ranks 8/16/32/64).
  - Realistic prompts derived from HumanEval and GSM8K subsets to mirror Databricks quality checks.
- **Metrics:** tokens/sec (prefill & decode), adapter swap latency, CPU utilisation per core, peak RSS, quality deltas vs. BF16 baseline.
- **Baselines:** `llama.cpp` CPU LoRA serving, `kTransformers` CPU backend, HybriMoE scheduling reference.

## Phase Plan
| Phase | Goals | Key Tasks |
| --- | --- | --- |
| P0 — Baseline | Validate measurement harness | Automate benchmark CLI, record per-metric dashboards, sanity-check adapter quality |
| P1 — Scheduling | Hide adapter overhead | Implement stream-style partitioning via async generation batches, experiment with adapter grouping inspired by Databricks PDL approach. |
| P2 — Quantization | Match quality while reducing latency | Integrate 8-bit/FP8 emulation on CPU (e.g., via bitsandbytes/int8) and evaluate hybrid attention strategies in software. |
| P3 — Kernel Optimisation | Target ≥10% throughput lead | Prototype fused LoRA matmuls using `torch.compile` with `inductor` CPU backend or custom C++ extensions; pursue SIMD-friendly layouts similar to Databricks’ row-wise scaling. |
| P4 — GPU Bridge | Prepare migration | Introduce optional CUDA path (vLLM-style) to validate portability and compare against Databricks figures. |

## Milestones
1. **M1:** Automated nightly benchmarks + baseline dashboards (owner: TBD, target: October 31, 2025).
2. **M2:** CPU runtime matches or exceeds llama.cpp by ≥10% tokens/sec on 4-adapter Zipf trace (target: November 7, 2025).
3. **M3:** Quality parity within ±0.25% on HumanEval/Math tasks under quantized serving (target: November 14, 2025).
4. **M4:** Publish CPU+GPU hybrid design doc aligned with Databricks multi-stream/PDL concepts (target: November 21, 2025).

## Open Questions
- Can adapter merging be amortized via shared low-rank caches without exhausting RAM?
- What level of batching is optimal before CPU dispatch becomes the bottleneck (as highlighted by Databricks’ CPU overhead findings)?
- Should we prototype thread pinning or NUMA-aware scheduling ahead of GPU integration?
