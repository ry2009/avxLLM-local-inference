# INFQ (Inference Quantization Container) — Specification v1

## Directory Layout
An INFQ package is a directory containing:

- `manifest.json` — metadata describing tensors, adapters, quantisation parameters, and binary payload locations.
- One or more binary files referenced by `data_file` fields in the manifest (for example `weights.bin`, `adapters.bin`).
- Optional auxiliary files (for example outlier buffers).

All multi-byte values use little-endian encoding. Offsets are byte offsets from the beginning of the referenced file.

## Manifest Schema (v1)
```json
{
  "version": 1,
  "endianness": "LE",
  "tensors": [
    {
      "name": "layers.0.attn.Wq",
      "dtype": "int8_rowwise",
      "rows": 4096,
      "cols": 4096,
      "block": 64,
      "scale_dtype": "fp16",
      "layout": "rowmajor_blocked",
      "data_file": "weights.bin",
      "offset_data": 0,
      "offset_scales": 4096 * 2,
      "outliers": {
        "dtype": "fp16",
        "data_file": "weights_outliers.bin",
        "offset": 0
      }
    }
  ],
  "adapters": [
    {
      "name": "adapter.foo",
      "rank": 32,
      "A": { "dtype": "int8_rowwise", ... },
      "B": { "dtype": "int8_rowwise", ... }
    }
  ]
}
```

### Required Fields
- `version`: must equal `1`.
- `endianness`: currently only `"LE"` (little-endian) is supported.
- `tensors`: array describing base model tensors.
- `adapters`: array describing LoRA-style adapters (optional; may be empty).

### Tensor Entry (`int8_rowwise`)
- `dtype`: quantised storage type. For v1 only `int8_rowwise` is supported.
- `rows`, `cols`: logical matrix shape (flattening higher dimensions into rows is allowed during conversion).
- `block`: number of elements per quantisation block (must be 64 for v1).
- `scale_dtype`: datatype used for per-row scale factors (`fp16`).
- `layout`: memory layout (`"rowmajor_blocked"`).
- `data_file`: file containing quantised weights.
- `offset_data`: byte offset of the first weight element within `data_file`.
- `offset_scales`: byte offset of the first scale value within `data_file` or a companion file.
- `outliers` (optional): metadata describing FP16 outliers stored as triples `(row:u32, col:u16, val:fp16)`;
  the object records `data_file`, `offset` (byte offset, 64-byte aligned), `count`, `record_bytes` (8) and
  `layout` (`"row:u32,col:u16,val:fp16"`).
- `outliers` (optional): location of higher-precision outlier values.

### Adapter Entry
- `name`: adapter identifier.
- `rank`: LoRA rank.
- `A`, `B`: tensor descriptors with the same schema as base tensors.

## Quantisation (Rowwise INT8)
For each row `r` with FP16/FP32 weights `w_r`:
1. Compute `scale_r = max(|w_r|) / 127`. If `scale_r < 1e-8`, clamp to `1e-8`.
2. Quantised values `q_r = round(w_r / scale_r)` clipped to `[-127, 127]`.
3. Store `q_r` as int8 and `scale_r` as FP16. Optional outliers may store values whose magnitude exceeded a configurable threshold (not yet emitted in v1).

## File Organisation Guidelines
- `weights.bin` typically stores all base tensors back-to-back. Scales may be co-located or written to a companion file if alignment is preferred.
- `adapters.bin` may interleave `A` and `B` matrices for each adapter to minimise seeks. Offsets in the manifest point to the start of each payload.

## Future Extensions
- Support for INT4 rowwise quantisation (`int4_rowwise`).
- KV-cache checkpoints.
- Checksums per tensor and optional compression flags.
- Perf summary metadata should capture host capabilities via fields such as
  `"flags"` (for example `["avx2", "avx512", "amx"]`), `"cpu_model"`, `"cores"`,
  and `"sdk"` so downstream dashboards can bucket results by hardware profile.
