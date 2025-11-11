# INFQ Quantization Pipeline

## Overview
INFQ (Inference Quantization Container) stores row-wise INT8 weights and FP16 scales with optional FP16 outliers. The runtime reads `manifest.json` as described in `formats/INFQ/spec.md` and expects all tensors to be pre-quantized before load.

## Conversion Workflow
1. Export base and adapter weights to safetensors.
2. Run the converter:
   ```bash
   python tools/convert/safetensors_to_infq.py \
     --base /path/base.safetensors \
     --adapter foo=/path/adapter.safetensors \
     --out /tmp/infq_dir \
     --outlier-threshold 6.0
   ```
   - `--outlier-threshold` controls FP16 outlier capture (set to `0` to disable).
3. Outputs include `manifest.json`, `weights.bin`, `adapters.bin`, and any outlier blobs. End-to-end validation lives in `python tests/infq_converter_test.py`.

## Runtime Path
- `CpuLoraEngine` currently dequantizes adapter tensors to FP32 on load (quantized execution remains future work).
- Base model tensors stay in packed row-wise INT8 form and feed directly into fused GEMM.

## Future Work
- Preserve adapters in quantized form through execution (Phase 3 follow-up).
- Add INT4/FP8 variants and richer manifest validation.
- Extend tests to cover multiple adapters and edge-case outlier layouts.

