#!/usr/bin/env python3
"""
Execute multiple benchmark runs via scripts/benchmark.py and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_str_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(v) for v in values]
    return [str(values)]


def build_command(run: Dict[str, Any], defaults: Dict[str, Any], metrics_path: Path) -> List[str]:
    root = Path(__file__).resolve().parent.parent
    benchmark_script = root / "scripts" / "benchmark.py"
    python_exec = Path(sys.executable)
    cmd: List[str] = [str(python_exec), str(benchmark_script)]

    def maybe_add(value: Any, option: str) -> None:
        if value is None:
            return
        cmd.append(option)
        cmd.append(str(value))

    merged: Dict[str, Any] = {**defaults, **run}

    cmd.extend(["--engine", merged.get("engine", "torch")])
    cmd.extend(["--base-model", merged["base_model"]])
    maybe_add(merged.get("num_requests"), "--num-requests")
    maybe_add(merged.get("zipf_alpha"), "--zipf-alpha")
    maybe_add(merged.get("seed"), "--seed")
    maybe_add(merged.get("max_new_tokens"), "--max-new-tokens")
    maybe_add(merged.get("iters"), "--iters")
    maybe_add(merged.get("warmup"), "--warmup")
    maybe_add(merged.get("temperature"), "--temperature")
    maybe_add(merged.get("top_p"), "--top-p")
    token_workers = merged.get("tokenize_overlap_workers")
    if token_workers:
        maybe_add(token_workers, "--tokenize-overlap-workers")

    if merged.get("do_sample"):
        cmd.append("--do-sample")

    if merged.get("include_base"):
        cmd.append("--include-base")

    if merged.get("profile") and merged.get("engine", "torch") == "torch":
        cmd.append("--profile")

    if merged.get("engine") == "llama_cpp":
        maybe_add(merged.get("n_ctx"), "--n-ctx")
        maybe_add(merged.get("n_threads"), "--n-threads")
        maybe_add(merged.get("lora_scale"), "--lora-scale")
        if merged.get("llama_verbose"):
            cmd.append("--llama-verbose")

    for adapter in ensure_str_list(merged.get("adapters")):
        if adapter:
            cmd.extend(["--adapter", adapter])

    prompts = merged.get("prompts")
    if prompts:
        for prompt in ensure_str_list(prompts):
            cmd.extend(["--prompt", prompt])

    cmd.extend(["--metrics-out", str(metrics_path)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple benchmark scenarios and aggregate results.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_sample.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_matrix.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--keep-metrics", action="store_true", help="Keep per-run metrics JSON files.")
    args = parser.parse_args()

    config = load_config(args.config)
    defaults: Dict[str, Any] = config.get("global", {})
    runs: List[Dict[str, Any]] = config.get("runs", [])
    if not runs:
        raise ValueError("No runs defined in configuration.")

    results: List[Dict[str, Any]] = []
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, run in enumerate(runs, start=1):
        name = run.get("name", f"run_{idx}")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            metrics_path = Path(tmp.name)
        command = build_command(run, defaults, metrics_path)
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parent.parent / "src"))
        start = time.perf_counter()
        if args.dry_run:
            print(f"[dry-run] {name}: {' '.join(command)} (metrics -> {metrics_path})")
            continue
        print(f"[run {idx}/{len(runs)}] {name}: executing benchmark...")
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - start
        run_record: Dict[str, Any] = {
            "name": name,
            "command": command,
            "elapsed_s": elapsed,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode != 0:
            print(f"[warn] {name} failed (exit {completed.returncode}). See stderr.")
            run_record["error"] = f"Command failed with exit code {completed.returncode}"
            results.append(run_record)
            if not args.keep_metrics and metrics_path.exists():
                metrics_path.unlink()
            continue

        if not metrics_path.exists():
            print(f"[warn] {name}: metrics file {metrics_path} missing.")
            results.append(run_record)
            continue

        with metrics_path.open("r", encoding="utf-8") as fh:
            run_record["metrics"] = json.load(fh)
        run_record["engine"] = run.get("engine", defaults.get("engine", "torch"))
        run_record["base_model"] = run.get("base_model", defaults.get("base_model"))
        if not args.keep_metrics:
            metrics_path.unlink(missing_ok=True)
        else:
            run_record["metrics_file"] = str(metrics_path)
        results.append(run_record)
        print(f"[done] {name}: tokens/sec={run_record['metrics']['tokens_per_second']:.2f}")

    summary = sorted(
        (
            {
                "name": r["name"],
                "engine": r.get("engine"),
                "tokens_per_second": r.get("metrics", {}).get("tokens_per_second"),
            }
            for r in results
            if "metrics" in r
        ),
        key=lambda item: item["tokens_per_second"] if item["tokens_per_second"] is not None else -1,
        reverse=True,
    )

    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "runs": results,
        "summary": summary,
    }
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2)
    print(f"[ok] wrote aggregated results to {args.output}")


if __name__ == "__main__":
    main()
