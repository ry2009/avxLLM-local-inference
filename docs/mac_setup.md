# macOS Setup Notes

These steps get the AVXLLM-local-inference toolchain running on Apple Silicon or
iNTEL Macs. They assume macOS 14+ and Homebrew.

## 1. Command line tools & Homebrew
```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 2. Toolchain packages
```bash
brew install cmake ninja llvm@17 python@3.12 rustup-init git pkg-config
```
- Add LLVM to your path (Apple Clang lacks some flags used by the project):
  ```bash
  echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
  echo 'export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"' >> ~/.zshrc
  echo 'export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"' >> ~/.zshrc
  ```
- Install Rust toolchain with `rustup-init -y --default-toolchain stable`.

## 3. Python environment
```bash
python3.12 -m venv ~/.virtualenvs/inf-eng
source ~/.virtualenvs/inf-eng/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 4. Build & test
```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFENG_AVX2=ON -DINFENG_ENABLE_PYTHON=ON
cmake --build build -j$(sysctl -n hw.logicalcpu)
ctest --test-dir build --output-on-failure
```

## 5. Quickstart scripts
- `scripts/run_local_inference.py` — downloads `sshleifer/tiny-gpt2` and optional adapters, then
  runs inference with full telemetry.
- `scripts/run_local_training.py` — fine-tunes a LoRA adapter on `data/distill_math.jsonl` samples.
- `scripts/run_local_pretrain.py` — builds a tiny causal LM checkpoint from `data/wiki_subset.jsonl`.
- `scripts/run_local_eval.py` — feeds `data/math_prompts.jsonl` through the runtime and saves telemetry.
- `scripts/run_end_to_end.py` — orchestrates download + LoRA training + evaluation (see `docs/quickstart_pipeline.md`).
- `scripts/download_manifest.py` — prefetch the assets listed in `configs/sample_assets.json` so demos work offline.
- `scripts/run_telemetry_matrix.py` — iterate through multiple adapters and export TPS/TTFT aggregates.

## 6. Makefile shortcuts
```bash
make setup        # create venv + install deps
make cpp-build    # configure + build native engine
make test         # run Catch2 + pytest suites
make run-infer    # execute the inference helper script with telemetry enabled
make run-eval     # run the dataset-driven eval script + telemetry export
make run-pipeline # end-to-end download/train/eval loop
make run-telemetry# gather TPS/TTFT matrix across adapters
make run-manifest # download the sample asset manifest
make check-mac    # run scripts/check_mac_env.py to confirm toolchain availability
```

Export `HF_TOKEN` (or use the `--token` flag) if your Hugging Face repos require authentication.
