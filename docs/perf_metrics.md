# Inference Metrics (TPS, TTPS, TTFT)

This note formalizes the latency/throughput numbers reviewers ask for and explains where to
instrument them in the Python reference runtime, the C++ engine, and the Rust tokenizer.

## Metric glossary
| Metric | Definition | How to compute |
| --- | --- | --- |
| **TPS (Tokens Per Second)** | Rate for a single request group. `new_tokens / latency_s` for that group. | Already emitted by `CpuPeftRuntime.benchmark` and the C++ scheduler traces (decode stage counter). |
| **TTPS (Total Tokens Per Second)** | Aggregate throughput across all concurrent requests. `sum(new_tokens)` divided by wall clock of the run. | Sum TPS across adapters or batches before dividing by elapsed pipeline time. Useful when multiple adapters interleave. |
| **TTFT (Time To First Token)** | Time from scheduler enqueue to the first decoded token leaving the sampler. Critical for interactive flows. | Capture clock at enqueue, subtract it from the timestamp emitted when decode kernel writes token #1. |
| **STIL (Scheduler Time In Loop)** | How long a batch remains inside the scheduler before dispatch. Shows queueing pressure. | Difference between `Scheduler::Stage::kEnqueue` and `kDispatch`. |
| **KV Residency** | Effective lifetime (s) a KV page stays pinned. Correlates with cache pressure. | Tracked via `mem::KvPagedArena::Snapshot`. |

## Python reference runtime (`src/peft_cpu_runtime`)
- `CpuPeftRuntime.generate` groups requests per adapter and can overlap tokenization. Wrap
  tokenization + generation loops with `time.perf_counter` to capture TTFT by inserting:
  ```python
  group_start = time.perf_counter()
  first_token_latency = None
  ...
  outputs = self.model.generate(...)
  if first_token_latency is None:
      first_token_latency = time.perf_counter() - group_start
  ```
- `benchmark()` already reports TPS and TTPS. When `_profiling_enabled` is set (via CLI `--telemetry`),
  it now records the minimum TTFT observed per iteration and surfaces it as `ttft_s` plus
  `avg_ttft_s` in the aggregate payload.

## C++ runtime (`engine/runtime`)
- Use `infeng::telemetry::ScopedCounter` to surround the following hot paths:
  - `Scheduler::EnqueueBatch` → `Scheduler::DispatchNext`: queue wait / STIL.
  - `CpuLoraEngine::Prefill` and `CpuLoraEngine::Decode`: per stage latency.
  - `tokenizer::EncodeBatch`: time for Rust tokenizer bridge; subtract from TTFT to isolate kernel cost.
- Emit counters to CSV via `telemetry::flush_csv` so CI artifacts land under `reports/*.csv`. Each row
  becomes `stage,elapsed_us`, making it trivial to compute medians for TPS/TTFT.

## Rust tokenizer (`engine/tokenizer`)
- The `fasttok` crate already exposes a C ABI. Add optional `tracing` spans guarded by the
  `FASTTOK_TRACING=1` env var to capture:
  - `Tokenizer::encode_stream_start`
  - `Tokenizer::encode_stream_next`
  - `Tokenizer::cache_lookup`
- Expose aggregate stats to C++ via a new `fasttok_get_metrics` function that returns
  `tokens_processed`, `avg_us_per_token`, and cache hit ratio. This keeps the “Rust + C++” conversation
  symmetric: Rust reports upstream prep costs; C++ reports downstream decode/adapter costs.

## Surfacing the metrics
1. Enable profiling in Python CLI: `peft-cpu-run infer ... --telemetry --telemetry-csv reports/runtime_metrics.csv`.
2. In native tests/benches, pass `INFENG_ENABLE_TELEMETRY=1` (CMake option TBD) or call
   `telemetry::flush_csv` from your harness.
3. Upload the CSVs as workflow artifacts (already wired for tokenizer/GEMM/attention jobs) so reviewers
   can diff TPS/TTPS/TTFT without reproducing locally.

For a deeper architectural overview, read `docs/engine_vision.md` and `docs/engine_roadmap.md`. This
file should accompany design conversations with top labs when walking them through latency numbers.
