# INF-ENG CPU Inference Engine — Architecture Vision

## Mission
Deliver a CPU-native inference engine that rivals GPU-first stacks (vLLM, sglang, TensorRT, llama.cpp) on throughput, quality, and developer ergonomics for smol-to-mid models. The engine must:
- Sustain competitive tokens/sec on commodity multi-core CPUs by fusing base and PEFT workloads.
- Support full-precision and quantized weights with minimal quality loss.
- Expose a clean API for batched, multi-tenant serving, prioritizing low-latency adapter hot-swaps.
- Remain self-contained: no external runtime dependencies beyond the system BLAS/oneDNN when it improves performance.

## Guiding Principles
1. **CPU-First Design**  
   Optimize for cache locality, SIMD utilization (AVX2/AVX-512/AMX), and NUMA awareness. Treat GPUs as optional accelerators that can be bolted on later.

2. **PEFT as a First-Class Workload**  
   LoRA/DoRA/Q-LoRA adapters must load, activate, and combine at trivial cost. Target near-zero overhead for multi-adapter, multi-request batches by fusing adapter GEMMs with base model kernels.

3. **Quantization Without Regret**  
   Pair aggressive quantization (int4/int8/FP8 hybrids) with calibration and hybrid attention so accuracy regresses <0.3% on standard evals.

4. **Composable Runtime**  
   Separate scheduling, memory management, and kernel execution layers. Ensure components can be swapped (e.g., a different attention kernel or tokenizer) without shaking the entire stack.

5. **Instrumentation Everywhere**  
   Bake profiling hooks into every layer (tokenizer, scheduler, kernels) so that performance regressions are visible immediately.

## Core Components
| Layer | Responsibility | Notes |
| --- | --- | --- |
| Frontend API | Accept batched requests (REST/gRPC/CLI). Normalize prompts, adapter selections, and sampling configs. | Initial phase targets CLI + Python API; network endpoints later. |
| Scheduler | Assemble microbatches, manage Zipfian adapter distributions, and orchestrate overlapped execution across CPU cores. | Inspired by vLLM’s paged attention scheduler; must support priority queues and backpressure. |
| Memory Manager | Owns KV cache, adapter weights, quantized tensors, and scratch buffers. Handles NUMA placement and page locking. | Explore static preallocation for deterministic latency. |
| Kernel Backend | Implements decode/prefill attention, fused base+LoRA GEMMs, normalization, activation, and quantization kernels. | Written in C++ (with optional Rust prototypes), exposing a C ABI for Python bindings. |
| Quantization Pipeline | Provides offline/online conversions to engine-native formats (blocked int4/int8, row-wise scales, hybrid FP8). | Must integrate with calibration data to preserve quality. |
| Profiler + Telemetry | Captures per-stage latencies, tokens/sec, CPU utilisation, cache miss rates. Exposes metrics to CLI, logs, dashboards. | Leverage Linux perf / VTune integration for deep dives. |

## Performance Targets (Initial Hardware: i9-9880H, 8C/16T)
- **Base decode throughput:** ≥2× llama.cpp on TinyLlama/TinyLlama-LoRA at 1k context, deterministic decoding.
- **LoRA overhead:** ≤5% slowdown when serving up to 4 adapters per batch (rank ≤32).
- **Latency:** <10s to prefill 4 × 1k-token prompts; decode ≥4 tokens/sec on average.
- **Quality:** HumanEval/GSM8K within 0.3% of baseline BF16 for TinyLlama adapters.

## Key Technical Bets
1. **Fused Base + Adapter Kernels**  
   Combine LoRA low-rank projections with base model matmuls in a single GEMM using custom-packed weights. Reduces separate kernel launches and memory traffic.

2. **Hybrid Attention with Fast Hadamard Transform**  
   Apply FHT + FP8/FP16 mix to shrink KV cache bandwidth, mimicking Databricks’ hybrid approach but tuned for CPU caches.

3. **Adaptive Microbatch Scheduling**  
   Partition cores between base decode and adapter tasks dynamically (programmable dependent launch analogue). Use SMT to mask memory stalls.

4. **Engine-Native Quantization Format**  
   Define a packed layout (e.g., [block size 64 × int4 + per-row scale]) that aligns with SIMD lanes. Provide conversion utilities from HF safetensors.

5. **Tokenizer Precompute & Streaming**  
   Keep tokenizer hot on dedicated threads, reuse prefix hashes, and stream tokens to decode loop to hide CPU prep overhead.

## Milestone Themes
1. **Foundations:** CLI harness, baseline kernels using existing libraries, basic scheduler.  
2. **Custom Kernels:** Replace bottlenecks with fused C++ kernels; introduce quantized formats.  
3. **Scheduling & Overlap:** Achieve near-zero LoRA overhead through stream partitioning and dependent launches.  
4. **Quality & Validation:** Integrate eval harness, autop-run regression tests, calibrate quantization.  
5. **Production Hardening:** Add persistence, multi-tenant isolation, metrics export, optional GPU bridge.

## Risks & Mitigations
- **Kernel Complexity:** C++ SIMDised kernels are error-prone. Mitigate with unit tests against reference PyTorch ops and use libraries (oneDNN) when they match our shapes.
- **Quantization Drift:** Aggressive quantization may degrade accuracy. Mitigate with calibration datasets and hybrid formats (store sensitive layers in higher precision).
- **Scheduler Deadlocks:** Complex overlap logic can cause stalls. Build simulation tests (dummy workload) to validate scheduling decisions before integrating kernels.
- **Developer Velocity:** Hand-rolled kernels slow iteration. Maintain Python fallbacks for rapid prototyping while optimizing hot paths incrementally.

## Immediate Next Actions
1. Draft kernel execution plan (attention, MLP, LoRA fusion) with required data layouts.
2. Define engine-native weight file format (manifest + tensor packing) and converter prototypes.
3. Build microbenchmark harness (Google Benchmark or custom) to evaluate matmul/attention prototypes.
4. Flesh out scheduler design (queues, core assignment) and integrate instrumentation hooks.
5. Identify minimal model (TinyLlama) + LoRA adapters as target baseline for continuous regression testing.
