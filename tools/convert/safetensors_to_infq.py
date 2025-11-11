#!/usr/bin/env python3
"""Convert safetensors weights/adapters into INFQ rowwise INT8 format."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from safetensors.numpy import load_file

BLOCK_SIZE = 64


@dataclass
class QuantResult:
  data: np.ndarray
  scales: np.ndarray
  outlier_rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint32))
  outlier_cols: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint16))
  outlier_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float16))


def reshape_rows(arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
  if arr.ndim == 1:
    return arr.reshape(1, -1), 1, arr.shape[0]
  rows = int(np.prod(arr.shape[:-1]))
  cols = arr.shape[-1]
  reshaped = arr.reshape(rows, cols)
  return reshaped, rows, cols


def quantize_rowwise_int8(weights: np.ndarray, outlier_threshold: float) -> QuantResult:
  rows, cols = weights.shape
  data = np.empty((rows, cols), dtype=np.int8)
  scales = np.empty(rows, dtype=np.float16)
  outlier_rows: list[np.ndarray] = []
  outlier_cols: list[np.ndarray] = []
  outlier_values: list[np.ndarray] = []
  for r in range(rows):
    row = weights[r]
    if outlier_threshold > 0.0:
      mask = np.abs(row) > outlier_threshold
      if mask.any():
        indices = np.nonzero(mask)[0].astype(np.uint16)
        outlier_rows.append(np.full(indices.shape, r, dtype=np.uint32))
        outlier_cols.append(indices)
        outlier_values.append(row[mask].astype(np.float16))
        row_for_scale = row.copy()
        row_for_scale[mask] = 0.0
      else:
        row_for_scale = row
    else:
      row_for_scale = row

    max_abs = float(np.max(np.abs(row_for_scale)))
    if max_abs < 1e-8:
      max_abs = 1e-8
    scale = max_abs / 127.0
    scales[r] = np.float16(scale)
    q = np.clip(np.rint(row_for_scale / scale), -127, 127).astype(np.int8)
    data[r] = q

  if outlier_rows:
    rows_concat = np.concatenate(outlier_rows)
    cols_concat = np.concatenate(outlier_cols)
    values_concat = np.concatenate(outlier_values)
  else:
    rows_concat = np.empty(0, dtype=np.uint32)
    cols_concat = np.empty(0, dtype=np.uint16)
    values_concat = np.empty(0, dtype=np.float16)

  return QuantResult(
      data=data,
      scales=scales,
      outlier_rows=rows_concat,
      outlier_cols=cols_concat,
      outlier_values=values_concat,
  )


def parse_adapter_spec(values: Iterable[str]) -> Dict[str, Path]:
  result: Dict[str, Path] = {}
  for spec in values:
    if "=" not in spec:
      raise SystemExit(f"adapter spec '{spec}' must be name=path")
    name, path_str = spec.split("=", 1)
    result[name.strip()] = Path(path_str.strip())
  return result


def write_tensor(out_file: Path, quant: QuantResult) -> Tuple[str, int, int]:
  out_file.parent.mkdir(parents=True, exist_ok=True)
  with out_file.open("ab") as fp:
    offset_data = fp.tell()
    fp.write(quant.data.tobytes(order="C"))
    offset_scales = fp.tell()
    fp.write(quant.scales.tobytes(order="C"))
  return str(out_file.name), offset_data, offset_scales


def write_outliers(out_file: Path, quant: QuantResult) -> Optional[Dict[str, int]]:
  if quant.outlier_rows.size == 0:
    return None
  out_file.parent.mkdir(parents=True, exist_ok=True)
  records = np.zeros(
      quant.outlier_rows.size,
      dtype=[("row", "<u4"), ("col", "<u2"), ("value", "<u2")],
  )
  records["row"] = quant.outlier_rows
  records["col"] = quant.outlier_cols
  records["value"] = quant.outlier_values.view(np.uint16)
  with out_file.open("ab") as fp:
    current = fp.tell()
    pad = (-current) % 64
    if pad:
      fp.write(b"\0" * pad)
    offset = fp.tell()
    fp.write(records.tobytes())
  return {
      "data_file": str(out_file.name),
      "offset": int(offset),
      "count": int(records.shape[0]),
      "record_bytes": 8,
      "layout": "row:u32,col:u16,val:fp16",
      "align": 64,
  }


def convert_base(base_path: Path, out_dir: Path, outlier_threshold: float) -> list:
  base_tensors = load_file(str(base_path))
  entries = []
  weights_file = out_dir / "weights.bin"
  weights_file.unlink(missing_ok=True)
  outliers_file = out_dir / "weights_outliers.bin"
  outliers_file.unlink(missing_ok=True)
  for name, array in base_tensors.items():
    reshaped, rows, cols = reshape_rows(array)
    quant = quantize_rowwise_int8(reshaped, outlier_threshold)
    data_file, offset_data, offset_scales = write_tensor(weights_file, quant)
    outlier_meta = write_outliers(outliers_file, quant)
    entries.append({
        "name": name,
        "dtype": "int8_rowwise",
        "rows": rows,
        "cols": cols,
        "block": BLOCK_SIZE,
        "scale_dtype": "fp16",
        "layout": "rowmajor_blocked",
        "data_file": data_file,
        "offset_data": offset_data,
        "offset_scales": offset_scales,
    })
    if outlier_meta:
      entries[-1]["outliers"] = outlier_meta
  return entries


def find_adapter_tensors(map_: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
  keys = {k.lower(): k for k in map_.keys()}
  key_a = keys.get("a") or keys.get("lora_a") or keys.get("adapter.a")
  key_b = keys.get("b") or keys.get("lora_b") or keys.get("adapter.b")
  if not key_a or not key_b:
    raise SystemExit("adapter safetensors must contain tensors 'A' and 'B'")
  return map_[key_a], map_[key_b]


def convert_adapters(specs: Dict[str, Path], out_dir: Path, outlier_threshold: float) -> list:
  if not specs:
    return []
  adapters_file = out_dir / "adapters.bin"
  adapters_file.unlink(missing_ok=True)
  outliers_file = out_dir / "adapters_outliers.bin"
  outliers_file.unlink(missing_ok=True)
  entries = []
  for name, path in specs.items():
    tensors = load_file(str(path))
    tensor_a, tensor_b = find_adapter_tensors(tensors)
    reshaped_a, rows_a, cols_a = reshape_rows(tensor_a)
    reshaped_b, rows_b, cols_b = reshape_rows(tensor_b)
    quant_a = quantize_rowwise_int8(reshaped_a, outlier_threshold)
    quant_b = quantize_rowwise_int8(reshaped_b, outlier_threshold)
    data_file_a, offset_data_a, offset_scales_a = write_tensor(adapters_file, quant_a)
    data_file_b, offset_data_b, offset_scales_b = write_tensor(adapters_file, quant_b)
    outlier_meta_a = write_outliers(outliers_file, quant_a)
    outlier_meta_b = write_outliers(outliers_file, quant_b)
    entries.append({
        "name": name,
        "rank": int(rows_b),
        "A": {
            "dtype": "int8_rowwise",
            "rows": rows_a,
            "cols": cols_a,
            "block": BLOCK_SIZE,
            "scale_dtype": "fp16",
            "layout": "rowmajor_blocked",
            "data_file": data_file_a,
            "offset_data": offset_data_a,
            "offset_scales": offset_scales_a,
        },
        "B": {
            "dtype": "int8_rowwise",
            "rows": rows_b,
            "cols": cols_b,
            "block": BLOCK_SIZE,
            "scale_dtype": "fp16",
            "layout": "rowmajor_blocked",
            "data_file": data_file_b,
            "offset_data": offset_data_b,
            "offset_scales": offset_scales_b,
        },
    })
    if outlier_meta_a:
      entries[-1]["A"]["outliers"] = outlier_meta_a
    if outlier_meta_b:
      entries[-1]["B"]["outliers"] = outlier_meta_b
  return entries


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert safetensors into INFQ")
  parser.add_argument("--base", type=Path, required=True, help="Base model safetensors file")
  parser.add_argument("--adapter", action="append", default=[], help="Adapter spec name=path")
  parser.add_argument("--out", type=Path, required=True, help="Output INFQ directory")
  parser.add_argument(
      "--outlier-threshold",
      type=float,
      default=6.0,
      help="Absolute value threshold for capturing FP16 outliers (0 disables outliers)",
  )
  args = parser.parse_args()

  out_dir = args.out
  out_dir.mkdir(parents=True, exist_ok=True)

  base_entries = convert_base(args.base, out_dir, args.outlier_threshold)
  adapter_entries = convert_adapters(parse_adapter_spec(args.adapter), out_dir, args.outlier_threshold)

  manifest = {
      "version": 1,
      "endianness": "LE",
      "tensors": base_entries,
      "adapters": adapter_entries,
  }

  manifest_path = out_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
