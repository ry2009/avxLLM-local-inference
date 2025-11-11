## Future Enhancements for macOS Intel Experiments
- Lightweight RLHF loop (reward model + PPO/SFT) with CPU-efficient batching and resumable checkpoints.
- Adapter composition lab (DoRA, IA3, full-rank adapters) with hot-swaps and latency profiling.
- MoE + mixture scheduling playground supporting sparsity-aware KV cache management.
- Quantization sweeps (int4/int8/FP8) with automated accuracy reports and Pareto frontier visualization.
- Knowledge distillation toolkit to compress larger teacher models into compact student LoRAs locally.
- ONNX / oneDNN backend bridge for cross-validation against alternative inference stacks.
- Automated eval harness (GSM8K, HumanEval, MTB benchmarks) with nightly dashboards.
- Prompt/trace simulator to stress-test memory footprints across retrieval-augmented workloads.

