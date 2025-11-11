# CPU PEFT Serving Initiative — Baseline Requirements

## Objectives
- Reproduce and extend fast PEFT (LoRA) serving techniques on commodity CPU hardware before introducing GPU-specific optimizations.
- Target a **≥10% throughput gain** over the best available open-source CPU LoRA serving baselines at comparable quality (identical adapters and quantization).

## Current Constraints
- **Host hardware:** Intel(R) Core(TM) i9-9880H @ 2.30GHz, 8 cores / 16 threads.
- **Memory:** 16 GB system RAM.
- **GPU availability:** none initially; design must degrade gracefully to CPU-only mode with a migration path to GPUs.
- **Software stack:** macOS with Python 3.12; no pre-existing project files in repo.

## Target Workloads
- **Base model:** prioritize models ≤3 B parameters to stay within memory limits (e.g., Mistral 2.4 B, Phi-3 mini) while maintaining relevance to enterprise LoRA use cases.
- **Adapters:** LoRA ranks 8–64 with Zipfian adapter request distributions to mimic production multi-tenant serving.
- **Latency targets:** maximize tokens/sec for both prefill-heavy and decode-heavy traces (context lengths 1 k–4 k).

## Candidate Baselines (CPU / Hybrid)
- **llama.cpp (GGML/GGUF):** mature CPU-first runtime with per-request LoRA adapter hot-swapping endpoints and GGUF quantization variants.
- **kTransformers:** Transformers-compatible runtime focused on hybrid CPU/GPU scheduling, AMX/NUMA optimizations, and rapid adapter experiments.
- **HybriMoE (kTransformers-based):** research prototype that improves CPU-GPU resource utilization for MoE/LoRA workloads via dynamic scheduling, providing a comparative ceiling for multi-adapter throughput.

## Measurement Framework
- Benchmark harness should capture: tokens/sec, p95 latency, CPU utilization, peak RSS, adapter swap latency.
- Use repeatable synthetic traces (e.g., multi-adapter Zipf distribution) plus at least one public evaluation suite (HumanEval, GSM8K slices) for quality audits.

## Immediate Next Steps
1. Validate baseline throughput on selected frameworks using identical LoRA adapters.
2. Profile CPU hotspots (kernel-level traces, Python overhead) to inform optimization roadmap.
3. Draft architecture for a unified serving loop supporting stream partitioning, adapter batching, and quantization experiments (now including llama.cpp GGUF runtime integration).
