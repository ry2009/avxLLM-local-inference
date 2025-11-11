#!/usr/bin/env bash
set -euo pipefail

echo "== COMPILERS =="; { clang --version || true; } ; { clang++ --version || true; }
echo "== BREW LLVM17 =="; if brew --prefix llvm@17 >/dev/null 2>&1; then BREW_LLVM=$(brew --prefix llvm@17); echo "$BREW_LLVM"; ls -1 "$BREW_LLVM/bin/clang++" || true; fi
echo "== SDKROOT =="; echo "${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)}"
echo "== CPU =="
sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null || true
echo "== FLAGS =="; sysctl -a 2>/dev/null | grep -i avx 2>/dev/null || true
echo "== PYTHON/NUMPY =="; python3 -c "import sys; print(sys.version)"; python3 -c "import numpy as np; print(np.__version__)" 2>/dev/null || echo "numpy missing"
echo "== CMAKE CACHE =="; [ -f build/CMakeCache.txt ] && grep -E 'INFENG_|CMAKE_CXX_COMPILER|OSX_SYSROOT' build/CMakeCache.txt || echo "no cache"
