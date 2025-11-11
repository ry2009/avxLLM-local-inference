# Throughput Snapshot (MacBook Pro M3 Max)

| Benchmark | Tokens/s (inf-eng) | Tokens/s (baseline) | Speedup | Notes |
| --- | --- | --- | --- | --- |
| attn_decode_int8_qk (1 batch, seq 1024) | 13.94M | 8.44M | 1.65× | Data: `reports/attn_decode_relwithdeb.csv` (baseline=llama.cpp fp16) |
| attn_decode_int8_qk (batch 4, seq 1024) | 13.89M | 8.52M | 1.63× | Same source as above |
| attn_prefill seq32_dim128 | 340K | 185K | 1.84× | `reports/attn_prefill_summary.json` |
| attn_prefill seq64_dim128 | 437K | 189K | 2.31× | same |
| attn_prefill seq128_dim128 | 456K | 183K | 2.50× | same |

**How to reproduce**
1. `make run-telemetry` to collect adapter TPS/TTFT metrics.
2. `python scripts/run_throughput_sweep.py --adapter base=` to sweep prompt lengths.
3. Compare against llama.cpp or vLLM runs using the same prompt lengths (see
   `docs/throughput_playbook.md`).
