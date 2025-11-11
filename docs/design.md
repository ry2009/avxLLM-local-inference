# INF-ENG Design Overview

Refer to `docs/engine_vision.md` for the architecture vision and `docs/engine_roadmap.md` for milestone planning. Detailed subsystem notes will live alongside implementation (e.g., `docs/scheduler.md`, `docs/quant.md`) as they are authored.

## Tokenizer Pipeline Summary

- Rust-backed streaming tokenizer (HF `tokenizers`) exposed via C ABI (`inf_tok_*`) with ABI versioning and explicit error codes.
- Prefix cache is keyed on the first `k` characters, LRU-managed with configurable capacity, and zero-copy batches are surfaced to C++ bindings until the next `encode_stream_next` call.
- Thread selection honors explicit overrides (`set_thread_override`) and the `INFENG_TOK_THREADS` environment variable before falling back to per-call hints.
- Benchmarks cover native vs Python tokenizers; CSV outputs (`reports/tokenizer*.csv`) feed the perf gate and CI artifacts.
