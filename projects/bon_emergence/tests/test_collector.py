from __future__ import annotations

from pathlib import Path


def test_placeholder():
    assert Path(__file__).exists()
