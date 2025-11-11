# Kernel Overlap Strategy Notes

## Goals
- Hide LoRA adapter overhead by overlapping adapter matmuls with base model decode steps.
- Port Databricks-style multiprocessor partitioning concepts to CPU threading and llama.cpp backends.
- Maintain quality parity while experimenting with quantized attention paths.

## Torch Runtime Ideas
- **Thread pools as pseudo streams:** dedicate a `ThreadPoolExecutor` for adapter matmuls and overlap token-level generation via `torch.compile` custom calls.
- **Adapter grouping:** batch requests by adapter rank and shape, enqueue grouped GEMMs so they can run concurrently with base model decode on separate Python threads (protected by `torch.set_num_threads` partitioning).
- **Prefetch + caching:** populate adapter weight tensors on pinned memory and reuse pre-tokenized inputs to cut CPU stalls.
- **Attention hybridization:** emulate FP8 + BF16 hybrid attention by mixing int8 GEMM kernels (via `bitsandbytes`) with high-precision softmax.

## llama.cpp Runtime Ideas
- **Adapter prebinding:** keep GGUF adapters resident and only adjust scaling, minimizing calls into C bindings.
- **Decode overlap:** spawn lightweight threads to prepare next prompt chunk while llama.cpp processes current token, leveraging its streaming API when available.
- **Dependent launch analogue:** queue `llama_set_adapter_lora` for the next request while current completion is still streaming, mirroring Programmatic Dependent Launch behavior.

## Cross-Cutting Tasks
1. Instrument detailed timelines (per-adapter tokenize vs. decode) using the new profiling hooks as a baseline.
2. Prototype shared ring buffers for adapter GEMM outputs so both runtimes can reuse host-resident results.
3. Validate that overlapping work does not regress quality by running HumanEval and GSM8K checkpoints on every major change.
4. Iterate on the thread-based tokenizer overlap prototype (current +2.5% throughput uplift) to overlap adapter matmuls next.
