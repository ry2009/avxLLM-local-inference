# INF-ENG Open Questions & Experiment Backlog

## Unknowns to Resolve
1. **Kernel Implementation Language**
   - C++20 with hand-written intrinsics vs. leveraging ISPC/Rust for SIMD generation.
   - Decision factors: developer velocity, tooling, ease of integration with Python bindings.

2. **Quantized Format Specification**
   - Block size, scale granularity (per-row vs. per-column), handling of outliers.
   - Need data on accuracy vs. storage cost for TinyLlama + LoRA adapters under different formats.

3. **KV Cache Strategy**
   - Balance between compression ratio and random access speed; whether to maintain mixed-precision caches.
   - Should we adopt a sliding window or paged attention analogue for long contexts?

4. **Tokenizer Integration**
   - Sticking with Hugging Face tokenizers vs. implementing a custom SIMD-accelerated tokenizer.
   - Requirements: streaming support, prefix caching, multi-threaded decoding.

5. **Scheduler Ergonomics**
   - API for expressing priorities, QoS tiers, and fairness across tenants.
   - Need to define SLA metrics (p99 latency, adapter swap time) and admission control.

6. **Adapter Lifecycle**
   - Format for the engine-native adapter artifacts, versioning, compatibility guarantees.
   - Hot reload semantics and garbage collection policy for inactive adapters.

7. **Evaluation Corpus**
   - Standard set of prompts/datasets for regression checks (HumanEval subset, GSM8K slice, domain-specific tasks).
   - Storage, licensing, and automation for nightly runs.

8. **Observability Stack**
   - Which metrics to surface (tokens/sec, latency, cache hit-rate, thread utilization) and how (Prometheus, OTLP).
   - Target integrations (Grafana dashboards, command-line summaries).

## Resource & Tooling Needs
- **Benchmark Hardware:** Additional CPUs with AVX-512/AMX capability; cloud credits for diverse architectures.
- **Instrumentation Tools:** Access to VTune, perf, flamegraph generators, cachegrind.
- **Testing Infrastructure:** CI runners with the necessary CPU features, plus storage for quantized artifacts.
- **Data Storage:** Repository for engine-native weight formats and regression datasets.
- **Developer Toolchain:** Compilers (clang/ICC), pybind11, Google Benchmark, profiling libraries.

## Immediate Experiments
1. **Matmul Microbenchmark**
   - Implement baseline C++ GEMM for TinyLlama dimensions (hidden 6144) and measure vs. MKL/oneDNN.
   - Objective: identify achievable speedups with fused LoRA paths.

2. **Hybrid Attention Prototype**
   - Write a Python/NumPy reference for FHT + FP8 mixing to validate numerical stability before C++ implementation.

3. **Quantization Sweep**
   - Convert TinyLlama weights to int8/int4 with various scale granularities; measure HumanEval/GSM8K delta.

4. **Scheduler Simulation**
   - Build a discrete-event simulator (Python) to test different scheduling strategies under Zipfian adapter loads.

5. **Tokenizer Overlap Extension**
   - Extend current threaded overlap prototype to pipeline adapter matmuls; collect detailed CPU utilization traces.

6. **Adapter Format Converter**
   - Draft a converter from Hugging Face safetensors LoRA to proposed engine-native format; evaluate load times.

7. **Telemetry Baseline**
   - Define the schema and implement initial CLIs that parse benchmark outputs into dashboards.

## Outstanding Decisions (Need Input)
- Prioritization between fused kernels vs. quantization pipeline—should we pursue both concurrently or sequence them?
- Target programming model for runtime (single binary with plugin interface vs. modular library).
- Long-term plan for community contribution: open-source vs. private repo with selective release.
