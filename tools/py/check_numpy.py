"""Lightweight NumPy import smoke test with JSON output."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")

try:
    import numpy as _np

    payload = {"ok": True, "ver": _np.__version__}
except Exception as exc:  # pragma: no cover - informational
    payload = {"ok": False, "error": repr(exc)}

json.dump(payload, sys.stdout)
sys.stdout.write("\n")
