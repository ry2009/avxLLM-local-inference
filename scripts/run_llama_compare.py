#!/usr/bin/env python3
"""Compare CpuPeftRuntime vs llama.cpp python on identical prompts."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download
from llama_cpp import Llama

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.utils import resolve_dtype


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CPU runtime vs llama.cpp")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", default="models/compare-base")
    parser.add_argument("--adapter-id")
    parser.add_argument("--adapter-name", default="compare")
    parser.add_argument("--adapter-dir", default="adapters/compare")
    parser.add_argument("--llama-model", required=True, help="Path to GGUF for llama-cpp")
    parser.add_argument("--prompts", default="data/math_prompts.jsonl")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--token")
    parser.add_argument("--revision")
    parser.add_argument("--out", default="reports/llama_compare.json")
    return parser.parse_args(argv)


def _download(repo_id: str, target: Path, token: Optional[str], revision: Optional[str]) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            revision=revision,
            token=token,
            resume_download=True,
        )
    )


def _load_prompts(path: Path, limit: int) -> List[str]:
    prompts: List[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            prompts.append(record.get("prompt") or record.get("text") or line)
        except json.JSONDecodeError:
            prompts.append(line)
        if len(prompts) >= limit:
            break
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def _runtime_metrics(runtime: CpuPeftRuntime, prompts: List[str], adapter: Optional[str], max_tokens: int) -> Dict[str, float]:
    batch = RequestBatch(
        requests=[InferenceRequest(prompt=p, adapter_name=adapter) for p in prompts],
        trace_config=InferenceTraceConfig(max_new_tokens=max_tokens),
    )
    runtime.enable_profiling(True)
    _ = runtime.generate(batch)
    return runtime.benchmark(batch, num_warmup=0, num_iters=1)


def _llama_metrics(llama_model: str, prompts: List[str], max_tokens: int, threads: int) -> Dict[str, float]:
    llm = Llama(model_path=llama_model, n_threads=threads, n_gpu_layers=0)
    total_tokens = 0
    start = time.perf_counter()
    for prompt in prompts:
        output = llm(prompt, max_tokens=max_tokens, temperature=0.0, top_p=0.95)
        text = output["choices"][0]["text"]
        total_tokens += len(text.split())
    elapsed = time.perf_counter() - start
    return {
        "tokens_per_second": total_tokens / elapsed if elapsed else 0.0,
        "avg_ttft_s": None,
        "outputs": len(prompts),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model_path = _download(args.model_id, Path(args.model_dir), token, args.revision)
    adapter_map = {}
    adapter_name = None
    if args.adapter_id:
        adapter_path = _download(args.adapter_id, Path(args.adapter_dir), token, args.revision)
        adapter_map[args.adapter_name] = str(adapter_path)
        adapter_name = args.adapter_name
    runtime = CpuPeftRuntime(
        base_model_id=str(model_path),
        adapter_map=adapter_map,
        torch_dtype=resolve_dtype(args.dtype),
        num_threads=args.threads,
    )
    prompts = _load_prompts(Path(args.prompts), args.limit)
    runtime_metrics = _runtime_metrics(runtime, prompts, adapter_name, args.max_new_tokens)
    llama_metrics = _llama_metrics(args.llama_model, prompts, args.max_new_tokens, args.threads)
    payload = {
        "runtime": runtime_metrics,
        "llama_cpp": llama_metrics,
        "prompts": prompts,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
