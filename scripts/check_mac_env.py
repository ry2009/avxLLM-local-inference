#!/usr/bin/env python3
"""Check macOS toolchain readiness for avxLLM-local-inference."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

REQUIREMENTS = {
    "cmake": "brew install cmake",
    "ninja": "brew install ninja",
    "llvm-strip": "brew install llvm",
    "cargo": "rustup-init -y",
    "python3": "brew install python@3.12",
}


@dataclass
class CheckResult:
    available: bool
    hint: str
    path: Optional[str]


def _check_command(cmd: str, hint: str) -> CheckResult:
    path = shutil.which(cmd)
    return CheckResult(available=path is not None, hint=hint, path=path)


def gather_status() -> Dict[str, CheckResult]:
    return {cmd: _check_command(cmd, hint) for cmd, hint in REQUIREMENTS.items()}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check macOS prerequisites for the repo.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    status = gather_status()
    missing = [cmd for cmd, res in status.items() if not res.available]
    summary = {
        cmd: {"available": res.available, "path": res.path, "hint": res.hint}
        for cmd, res in status.items()
    }
    summary["platform"] = platform.platform()

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for cmd, res in status.items():
            icon = "✅" if res.available else "❌"
            path = res.path or "not found"
            print(f"{icon} {cmd}: {path}")
            if not res.available:
                print(f"   hint: {res.hint}")
        print(f"Platform: {summary['platform']}")

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
