# Benchmark Workflow

## Toolchain Setup (clang-17 + cargo)

- **Shared requirements:** LLVM 17 toolchain (`clang`, `clang++`, `lld`), Rust `cargo`, Python 3.10+. The CMake presets expect these on `PATH`.
- **macOS (Homebrew LLVM + Command Line Tools):**
  1. `brew install llvm@17` and ensure `$(brew --prefix llvm@17)/bin` precedes `/usr/bin` in `PATH`.
  2. Keep Command Line Tools current: `softwareupdate --list | grep -i "Command Line Tools"` then `sudo softwareupdate -i "Command Line Tools for Xcode <version>"`.
  3. If the update fails, reinstall CLT cleanly:
     ```bash
     sudo rm -rf /Library/Developer/CommandLineTools
     xcode-select --install  # chooses the latest CLT
     sudo xcode-select -s /Library/Developer/CommandLineTools
     export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
     ```
  4. Configure the environment before running CMake:
     ```bash
     BREW_LLVM="$(brew --prefix llvm@17)"
     export CC="$BREW_LLVM/bin/clang"
     export CXX="$BREW_LLVM/bin/clang++"
     export CPPFLAGS="-I$BREW_LLVM/include -isysroot $SDKROOT"
     export LDFLAGS="-L$BREW_LLVM/lib -L$BREW_LLVM/lib/c++ \
       -Wl,-rpath,$BREW_LLVM/lib/c++ -Wl,-syslibroot,$SDKROOT"
     ```
  5. **Quick workaround for `!tapi-tbd` linker errors:** switch to LLVM’s `lld`:
     ```bash
     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
       -DCMAKE_OSX_SYSROOT="$SDKROOT" \
       -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
       -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
       -DINFENG_AVX2=ON -DINFENG_AVX512=ON
     cmake --build build -j
     ```
- **Ubuntu (22.04+):**
  ```bash
  sudo apt update
  sudo apt install -y clang-17 lld-17 cargo
  sudo update-alternatives --install /usr/bin/cc cc /usr/bin/clang-17 100
  sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-17 100
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 \
    -DINFENG_AVX2=ON -DINFENG_AVX512=ON
  ```
- After configuring either platform, rebuild and test:
  ```bash
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```

## Build & Run Commands

```bash
# Configure (Release, AVX-512)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/clang17.cmake -DINFENG_AVX512=ON

cmake --build build -j

# Run benches (CSV to reports/)
./build/tools/bench/bench_gemm_base   --benchmark_format=csv > reports/gemm_base.csv
./build/tools/bench/bench_gemm_lora   --benchmark_format=csv > reports/gemm_lora.csv
./build/tools/bench/bench_attn_decode --benchmark_format=csv > reports/attn_decode.csv
./build/tools/bench/bench_tokenizer   --benchmark_format=csv > reports/tokenizer.csv

# Convert to INFQ
python tools/convert/safetensors_to_infq.py --in /models/tinyllama --out /models_infq/tinyllama_infq

# Smoke serve (prints tokens/sec + p95)
python apis/python/serve_cli.py --model /models_infq/tinyllama_infq --adapters /adapters_infq/* --mode throughput --report reports/run.json
```

## Tokenizer Throughput & Baselines

```bash
# Native tokenizer (Google Benchmark + CSV)
./build/tools/bench/infeng_bench_tokenizer_gb \
  --model tests/data/test_tokenizer.json \
  --input_file tools/bench/prompts.txt \
  --threads=8 --prefix_k=128 --prefix_cache_entries=32

# Python baseline (HF tokenizers)
python tools/bench/python_tokenizer_baseline.py \
  --model tests/data/test_tokenizer.json \
  --input_file tools/bench/prompts.txt \
  --threads 8 \
  --output reports/tokenizer_py.csv

# Compare & enforce ≥5× speedup (defaults to TOKENIZER_MIN_SPEEDUP=5.0)
python tools/bench/compare_tokenizers.py \
  --native reports/tokenizer.csv \
  --python reports/tokenizer_py.csv \
  --baseline tools/perf/tokenizer_baseline.json

# Full perf sweep (1/2/4/8 threads + summary JSON)
tools/perf/run_tokenizer_perf.sh tests/data/test_tokenizer.json tools/bench/prompts.txt

# GEMM LoRA sweep (writes reports/gemm_lora.csv + gemm_summary.json)
tools/perf/run_gemm_perf.sh

# Attention prefill sweep (writes reports/attn_prefill.csv + attn_prefill_summary.json)
tools/perf/run_attn_perf.sh
```

### Baseline management

Each perf lane compares freshly measured CSVs against the JSON baselines in
`tools/perf/`. To refresh those baselines after kernel or hardware changes:

```bash
# GEMM baseline calibration
tools/perf/run_gemm_perf.sh --out reports/gemm_lora.csv
python tools/perf/compare_gemm.py --csv reports/gemm_lora.csv \
  --baseline-out tools/perf/gemm_baseline.json --update-baseline

# Attention prefill baseline calibration
tools/perf/run_attn_perf.sh --out reports/attn_prefill.csv
python tools/perf/compare_attn_prefill.py --csv reports/attn_prefill.csv \
  --baseline-out tools/perf/attn_prefill_baseline.json --update-baseline

# Tokenizer (existing flow)
tools/perf/run_tokenizer_perf.sh tests/data/test_tokenizer.json tools/bench/prompts.txt
```

Baseline files record the measured `tokens_per_s` along with a `_meta` header
(hardware model, core count, timestamp). CI uses the committed baselines; only run the
calibration flow above when intentionally updating them on a known-good machine.

### Required CI gates

Mark the following workflow runs as required in GitHub's branch-protection settings so perf regressions block merges:

- `tokenizer-perf` (≥5× native speedup vs. HF baseline)
- `gemm-perf` (fused GEMM throughput, LoRA tax)
- `attn-prefill-perf` (prefill speedup + perplexity guard)

# Additional perf sweeps

```bash
# Attention decode sweep (writes reports/attn_decode.csv + attn_decode_summary.json)
tools/perf/run_attn_decode.sh

# Scheduler Zipf synthetic sweep (writes reports/sched_zipf.csv + sched_zipf_summary.json)
tools/perf/run_sched_zipf.sh

# Example with explicit knobs
tools/perf/run_sched_zipf.sh \
  --load_rps 24 \
  --duration_s 90 \
  --mean_adapters 4 \
  --decode_threads 6 \
  --adapter_threads 4 \
  --pin_cores
```

The attention decode CSV now captures both fused and FP16 baselines with columns for
`tokens_per_s`, `baseline_tokens_per_s`, `perplexity_delta`, `scratch_ratio`,
`tile_tokens`, `acc_width`, `has_avx2`, `has_vnni`, `csr_outliers`, and
`l2_bytes_per_token` so we can diagnose tiling and cache usage directly from the log.
`compare_attn_decode.py` enforces:

- ≥1.15× speedup for ctx ≥ 2k tokens
- |Δ perplexity| ≤ 0.3 pp
- L2 diff ≤ 1e-3 (per-element, normalised)
- Scratch usage ≤ 1.25× the FP16 baseline

Kernel dispatch selects AVX-512+VNNI when available, otherwise falls back to
AVX2 and finally the scalar reference path.

The perf runner accepts `--tile_tokens=<N>` to override the tiling heuristic and
`--force_path={auto,scalar,avx2,avx512}` to pin the decode kernel path for
reproduction.

### Scheduler Zipf CSV schema

`sched,alpha,adapters_N,mean_adapters_per_req,rank,load_rps,duration_s,decode_threads,adapter_threads,pin_cores,p50_ms,p95_ms,base_p95_ms,lora_tax_pct,hot_reload_ms_p95,tokens_per_s,queue_ms_p95,decode_util,adapter_util,overlap_ratio`

- `load_rps` is the observed request rate over the measured window.
- `hot_reload_ms_p95` captures tail latency for simulated adapter reloads.
- `queue_ms_p95` reports the decode queue wait (request queue) 95th percentile.
- `decode_util`/`adapter_util` indicate the fraction of wall-clock time each stage stayed busy.
- `overlap_ratio` shows how often decode and adapter work ran simultaneously (1.0 = always overlapping).

`tools/perf/compare_sched_zipf.py` tracks throughput regressions against the stored baseline
and emits warnings when LoRA tax exceeds 10% or queue wait drifts beyond 200 ms.

**Tokenizer note:** CI currently enforces a temporary ≥3.5× speedup gate on AVX2 hosts
while tracking the long-term ≥5× target in the summary JSON.

To refresh decode and scheduler baselines:

```bash
# Attention decode baseline calibration
tools/perf/run_attn_decode.sh --out reports/attn_decode.csv
python tools/perf/compare_attn_decode.py --csv reports/attn_decode.csv \
  --baseline-out tools/perf/attn_decode_baseline.json --update-baseline

# Scheduler Zipf baseline calibration
tools/perf/run_sched_zipf.sh --out reports/sched_zipf.csv
python tools/perf/compare_sched_zipf.py --csv reports/sched_zipf.csv \
  --baseline-out tools/perf/sched_zipf_baseline.json --update-baseline
```
