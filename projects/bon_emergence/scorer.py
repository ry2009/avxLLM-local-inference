from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


def _normalise(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _safe_eval(expr: str) -> Optional[float]:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    structural = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant):
            if not isinstance(sub.value, (int, float)):
                return None
            continue
        if not isinstance(sub, structural):
            return None
    try:
        value = eval(compile(node, "<expr>", "eval"))
        return float(value)
    except Exception:
        return None


@dataclass
class ScoreResult:
    correct: bool
    details: Dict[str, float | str]


def exact_match(answer: str, completion: str) -> ScoreResult:
    return ScoreResult(
        correct=_normalise(answer) == _normalise(completion),
        details={},
    )


def numeric_tolerance(answer: str, completion: str, tolerance: float = 1e-6) -> ScoreResult:
    gold = _safe_eval(answer)
    guess = _safe_eval(completion)
    if gold is None or guess is None:
        return ScoreResult(False, {"gold": gold if gold is not None else "NaN", "guess": guess if guess is not None else "NaN"})
    return ScoreResult(abs(gold - guess) <= tolerance, {"gold": gold, "guess": guess, "tolerance": tolerance})


def registry() -> Dict[str, Callable[[str, str], ScoreResult]]:
    return {
        "exact": exact_match,
        "numeric": lambda answer, completion: numeric_tolerance(answer, completion),
        "mc": multiple_choice,
    }


def load_scorer(name: str) -> Callable[[str, str], ScoreResult]:
    try:
        return registry()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown scorer '{name}'. Available: {list(registry().keys())}") from exc


def pass_at_k(mask: Iterable[bool], k: int) -> float:
    successes = 0
    total = 0
    for idx, passed in enumerate(mask):
        total += 1
        if idx < k and passed:
            successes = 1
            break
    return float(successes)


CHOICE_PATTERN = re.compile(r"([A-Z])")


def _extract_choice(text: str) -> Optional[str]:
    match = CHOICE_PATTERN.search(text.upper())
    if match:
        letter = match.group(1)
        if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return letter
    return None


def multiple_choice(answer: str, completion: str) -> ScoreResult:
    expected = answer.strip().upper()
    choice = _extract_choice(completion)
    return ScoreResult(choice == expected, {"choice": choice or ""})


__all__ = [
    "ScoreResult",
    "exact_match",
    "numeric_tolerance",
    "multiple_choice",
    "load_scorer",
    "pass_at_k",
]
