# Rust Speedups

The tokenizer bridge in `engine/tokenizer` is written in Rust and compiled via
Cargo during the CMake build. To track its impact:

1. `tools/perf/run_tokenizer_perf.sh tests/data/test_tokenizer.json tools/bench/prompts.txt`
   generates `reports/tokenizer.csv` with tokens/sec per thread.
2. `python scripts/summarize_tokenizer_perf.py` reads that CSV and reports the
   optimal thread configuration. Example output:
   ```
   {'best_threads': 4, 'best_tokens_per_s': 28000.0, ...}
   ```
3. The Rust tokenizer benchmarking harness matches the LRU cache and prefix
   streaming strategy documented in `docs/engine_vision.md`.

To rerun the lower-level Rust benches directly:
```bash
cd engine/tokenizer
cargo bench
```
