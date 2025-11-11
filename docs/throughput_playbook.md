# Throughput Playbook

Steps to maximize local CPU inference performance on Mac/AVX hosts.

1. **Pin threads**
   - Set `INFENG_NUM_THREADS=<logical cores>` before running any runtime-based
     scripts, or pass `num_threads` when instantiating `CpuPeftRuntime`. The
     runtime now calls `torch.set_num_threads` and records the configured value
     so you can inspect it later. For interop pools, set
     `INFENG_NUM_INTEROP_THREADS` or pass `num_interop_threads`.

2. **Cache prompts**
   - Export `INFENG_TOKEN_CACHE=<entries>` (or pass `token_cache_size`) to avoid
     re-tokenizing identical prompts during sweeps/CI smoke tests.

2. **Warm caches**
   - Run `make run-manifest` to download the sample model/adapters.
   - Run `make run-ci-smoke` once; this primes tokenizer caches and the HF
     cache directories.

3. **Measure baselines**
   - `make run-telemetry` (or invoke `scripts/run_telemetry_matrix.py`) to gather
     TPS/TTFT across adapters.
   - `python scripts/run_throughput_sweep.py --adapter base= --lengths 32,64,128`
     writes `reports/throughput_sweep.json` with tokens/sec per prompt length.
   - `python scripts/run_prompt_benchmark.py --model-id ... --prompts data/math_prompts.jsonl`
      produces per-prompt CSV/JSON for labs that want fine-grained diffs.

4. **Compare changes**
   - Keep `reports/telemetry_matrix.json` and `reports/throughput_sweep.json`
     under version control (or linked in PRs) to prove throughput gains.

5. **Advanced**
   - Use `INFENG_NUM_THREADS` + `taskset`/`caffeinate` to pin the runtime to
     performance cores only.
   - Combine `scripts/run_throughput_sweep.py` with different `--adapter`
     settings to evaluate adapter fusion overheads.
   - `python scripts/run_llama_compare.py --model-id ... --llama-model path.gguf`
     → quick apples-to-apples tokens/sec comparison vs llama.cpp.
