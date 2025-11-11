from __future__ import annotations

from projects.bon_emergence.scorer import (
    exact_match,
    numeric_tolerance,
    load_scorer,
    pass_at_k,
)


def test_exact_match():
    assert exact_match("Paris", " paris ").correct
    assert not exact_match("Paris", "London").correct


def test_numeric_tolerance():
    res = numeric_tolerance("3.14159", "3.1416", tolerance=1e-3)
    assert res.correct
    assert not numeric_tolerance("2", "five").correct


def test_registry_lookup():
    scorer = load_scorer("exact")
    assert scorer("a", "a").correct


def test_pass_at_k():
    mask = [False, False, True, False]
    assert pass_at_k(mask, 1) == 0.0
    assert pass_at_k(mask, 3) == 1.0
