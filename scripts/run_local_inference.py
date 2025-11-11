#!/usr/bin/env python3
"""Plug-and-play inference runner for macOS/Linux workstations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import snapshot_download

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.utils import resolve_dtype


def _download(repo_id: str, target: Path, token: Optional[str], revision: Optional[str] = None) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        revision=revision,
        token=token,
        resume_download=True,
    )
    return Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local CPU inference with optional LoRA adapters.")
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2", help="HF repo id for the base model")
    parser.add_argument("--model-dir", default="models/quickstart-base", help="Cache directory for the base model")
    parser.add_argument("--adapter-id", help="Optional HF repo id for a LoRA/PEFT adapter")
    parser.add_argument("--adapter-name", default="quickstart", help="Logical adapter name")
    parser.add_argument("--adapter-dir", default="adapters/quickstart", help="Cache directory for the adapter")
    parser.add_argument("--prompts", nargs="*", default=["Write a limerick about AVX kernels."], help="Prompts to decode")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="float32", help="Model dtype (float32, bf16, fp16)")
    parser.add_argument("--token", help="HF token (falls back to HF_TOKEN env var)")
    parser.add_argument("--revision", help="Optional git revision for model/adapter")
    parser.add_argument("--telemetry", action="store_true", help="Emit TPS/TTPS/TTFT via runtime profiling")
    parser.add_argument("--benchmark-iters", type=int, default=1, help="Number of iterations for benchmark output")
    parser.add_argument("--benchmark-warmup", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    model_path = _download(args.model_id, Path(args.model_dir), token, args.revision)
    adapter_map: Dict[str, str] = {}
    if args.adapter_id:
        adapter_path = _download(args.adapter_id, Path(args.adapter_dir), token, args.revision)
        adapter_map[args.adapter_name] = str(adapter_path)

    runtime = CpuPeftRuntime(
        base_model_id=str(model_path),
        adapter_map=adapter_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    if args.telemetry:
        runtime.enable_profiling(True)

    requests = [
        InferenceRequest(prompt=prompt, adapter_name=args.adapter_name if adapter_map else None)
        for prompt in args.prompts
    ]
    batch = RequestBatch(
        requests=requests,
        trace_config=InferenceTraceConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        ),
    )
    outputs = runtime.generate(batch)
    result = {"prompts": args.prompts, "outputs": outputs}
    print(json.dumps(result, indent=2))

    if args.telemetry:
        metrics = runtime.benchmark(batch, num_warmup=args.benchmark_warmup, num_iters=args.benchmark_iters)
        print("\nTelemetry")
        print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
