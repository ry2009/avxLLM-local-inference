# CI Setup Guide

This repository ships a multi-job workflow (`.github/workflows/build.yml`) that compiles the
C++ engine, runs unit tests, and executes the tokenizer/GEMM/attention perf harnesses. Public
GitHub runners can execute everything as-is, but you should provision the following secrets to
avoid rate limits when tests or download scripts touch Hugging Face Hub:

| Secret | Why it matters | Scope |
| --- | --- | --- |
| `HF_TOKEN` | Read token for Hugging Face Hub. Used by download scripts and future tests that fetch public or gated checkpoints. | Repo-level Actions secret. Provide a READ role token from https://huggingface.co/settings/tokens |

The workflow automatically exports `HF_TOKEN` as both `HF_TOKEN` and
`HUGGINGFACE_HUB_TOKEN`, so any step (Python CLI, C++ tests invoking the tokenizer) can access
it without extra wiring. If the secret is absent, jobs continue with anonymous access.

## Runner cache directories
- `HF_HOME` is pinned to `${{ github.workspace }}/.hf-cache` so Hugging Face downloads are cached
  between jobs inside a workflow run.
- CMake/Ninja artifacts remain in `${{ github.workspace }}/build` per job.

## Local verification
Before pushing, run the same steps the workflow performs:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DINFENG_AVX2=ON -DINFENG_ENABLE_PYTHON=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

If you rely on private or rate-limited Hugging Face repos locally, export `HF_TOKEN` in your shell
or pass `--token` to `tools/download_assets.py` so commits stay reproducible.
