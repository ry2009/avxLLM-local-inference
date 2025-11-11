# INF-ENG CPU Inference Engine — Roadmap (Draft)

## Phase Overview
| Phase | Target Date | Focus | Key Deliverables |
| --- | --- | --- | --- |
| P0 Foundations | 2025-09-31 | Establish scaffolding, ensure measurable baselines | Microbenchmark harness, reference Python runtime, telemetry hooks |
| P1 Custom Kernels | 2025-10-12 | Replace hot paths with fused CPU kernels | Fused base+LoRA GEMMs, hybrid attention prototype, engine-native weight format |
| P2 Scheduling & Overlap | 2025-10-23 | Achieve low-overhead multi-adapter serving | Adaptive scheduler, SM/SMT partitioning analogue, tokenizer streaming |
| P3 Quantization & Quality | 2025-11-07 | Deliver aggressive quantization with minimal loss | Int4/int8/FP8 pipeline, calibration tooling, quality regression suite |
| P4 Production Hardening | 2025 final sprint | Operational readiness & extensibility | API layer (Python + REST), persistent cache, observability, optional GPU bridge |

## Phase Details

### P0 — Foundations (Now → 31 Oct 2025)
- **Microbenchmark Harness:** Google Benchmark / custom C++ harness measuring GEMM, attention, tokenizer throughput.
- **Reference Runtime:** Python-only baseline (current `CpuPeftRuntime`) refactored into architecture-aligned modules.
- **Telemetry Backbone:** Standard JSON/CSV export for tokens/sec, latency, utilization; integrate Plotly dashboard script.
- **CI Smoke Tests:** TinyLlama decode, LoRA adapter activations, basic unit tests for scheduler queues.

### P1 — Custom Kernels (1 Nov → 21 Nov 2025)
- **Fused Base+LoRA GEMM:** C++ SIMD kernel for (W + ΔW) x x with rank-aware packing; fallback to BLAS.
- **Hybrid Attention:** CPU kernel implementing FHT + FP8/FP16 mix; validate speed vs. PyTorch attention.
- **Data Layout Spec:** Define engine-native tensor packing (header, per-layer metadata, scale tensors).
- **Adapter Loader:** Load adapters into fused format; support rank 8–64, hot-swapping across batches.

### P2 — Scheduling & Overlap (22 Nov → 12 Dec 2025)
- **Core Partitioning:** Dynamic assignment of cores to decode vs. adapter streams; respect SMT pairs.
- **Dependent Launch Analogue:** Pre-stage adapter weights and inputs while current batch decodes.
- **Tokenizer Streaming:** Dedicated tokenizer threads with prefix caching and streaming into decode loops.
- **Latency SLA Tests:** Stress tests with Zipfian adapter distributions to confirm <5% LoRA overhead.

### P3 — Quantization & Quality (13 Dec 2025 → 9 Jan 2026)
- **Quant Pipeline:** Offline converter for BF16 → (int4/int8 + per-row scales) with metadata manifest.
- **Calibration & Eval:** Integrate small evaluation suite (HumanEval 0.25 subset, GSM8K slice) and diff reports.
- **KV Cache Compression:** Apply hybrid attention quantization to KV cache; ensure negligible perplexity drift.
- **Quality Dashboard:** Automated report tracking accuracy vs. baseline, surfaced alongside throughput metrics.

### P4 — Production Hardening (10 Jan → 6 Feb 2026)
- **API & Runtime Service:** Python client + REST endpoint with batching, retries, backpressure.
- **Persistence & Warmup:** Snapshot engine-native weights/adapters; pre-warm caches across restarts.
- **Observability:** Expose Prometheus metrics, structured logs, optional tracing.
- **GPU Bridge (Stretch):** Optional CUDA execution backend for cross-validation, ensuring architecture extensibility.

## Milestone Tracking
| Milestone | Description | Owner | Status |
| --- | --- | --- | --- |
| M0 | Engine vision & roadmap approved | TBD | Draft |
| M1 | Microbenchmark harness + ref runtime passing smoke tests | 2025-10-31 | Pending |
| M2 | Fused GEMM & hybrid attention outperform baselines | 2025-11-21 | Pending |
| M3 | Scheduler achieves ≤5% LoRA overhead (Zipf trace) | 2025-12-12 | Pending |
| M4 | Quant pipeline within 0.3% accuracy delta | 2026-01-09 | Pending |
| M5 | API + observability + persistence integrated | 2026-02-06 | Pending |

## Dependencies & Tooling
- **Languages:** C++20 (kernels, runtime), Python 3.12 (bindings/tooling), possible Rust prototypes.
- **Libraries:** oneDNN (optional), Google Benchmark, Catch2/GoogleTest, pybind11 for bindings.
- **Hardware Lab:** Baseline i9-9880H, target additional CPU platforms (AMD Zen 4, Intel Sapphire Rapids with AMX).

## Reporting Cadence
- Weekly progress update (tokens/sec, latency, accuracy, kernel coverage).
- Monthly milestone review with decision to proceed to next phase.
- Continuous benchmarking pipeline (nightly) once P1 assets land.
