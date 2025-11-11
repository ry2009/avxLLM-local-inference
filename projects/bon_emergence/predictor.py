from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _iter_run_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.glob("*.jsonl")):
        yield candidate


def _load_records(paths: Iterable[Path]) -> List[dict]:
    records: List[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    if not records:
        raise SystemExit("No BoN records found")
    return records


def _feature_vector(record: dict) -> Tuple[np.ndarray, int]:
    metrics = record["metrics"]
    pass_at = metrics["pass_at"]
    base = float(pass_at.get("1", 0.0))
    higher = [float(v) for k, v in pass_at.items() if k != "1"]
    best = max([base] + higher)
    delta = best - base
    unique_frac = float(metrics.get("unique_frac", 0.0))
    entropy = float(metrics.get("entropy", 0.0))
    avg_len = float(metrics.get("avg_completion_length", 0.0))
    features = np.array([base, delta, unique_frac, entropy, avg_len], dtype=np.float64)
    label = 1 if delta > 0.0 else 0
    return features, label


def fit_model(records: List[dict]):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise SystemExit("scikit-learn is required (pip install scikit-learn)") from exc

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for record in records:
        features, label = _feature_vector(record)
        X_list.append(features)
        y_list.append(label)
    X = np.vstack(X_list)
    y = np.array(y_list)

    if len(set(y.tolist())) < 2:
        raise ValueError("Need at least two classes in labels to fit logistic regression. Collect more runs.")

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    preds = model.predict(X)
    accuracy = (preds == y).mean()
    print(f"Training accuracy on {len(y)} samples: {accuracy:.3f}")
    return model


def export_model(model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        pickle.dump(model, fp)
    print(f"Saved predictor to {out_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit BoN emergence predictor from collected runs")
    parser.add_argument("--runs", type=Path, required=True, help="Directory or JSONL file with collector output")
    parser.add_argument("--out", type=Path, default=Path("reports/bon_runs/bon_predictor.pkl"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_files = list(_iter_run_files(args.runs))
    records = _load_records(run_files)
    model = fit_model(records)
    export_model(model, args.out)


if __name__ == "__main__":
    main()
