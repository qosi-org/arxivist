#!/usr/bin/env python
"""
evaluate.py — load a trained model + test set, print/save the confusion
matrix and accuracy.

Usage:
    python evaluate.py --results-dir results/6.1 --experiment 6.1
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained experiment")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--out", type=str, default="results/")
    p.add_argument(
        "--as-percentage",
        action="store_true",
        help="Report confusion matrix as row percentages (paper's Section 6 convention); "
        "default is absolute counts (paper's Section 5 convention)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    pkl_path = results_dir / "model_and_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} not found. Run train.py --experiment {args.experiment} first.")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    clf = data["clf"]
    X_test, y_test, class_names = data["X_test"], data["y_test"], data["class_names"]

    cm = clf.confusion_matrix(X_test, y_test, class_names, as_percentage=args.as_percentage)
    print(f"\n=== Experiment {args.experiment} ===")
    print(f"Train accuracy: {data['train_acc']:.4f}")
    print(f"Test accuracy:  {data['test_acc']:.4f}")
    print("\nConfusion matrix:")
    print(cm.to_string())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"confusion_matrix_{args.experiment}.csv"
    cm.to_csv(csv_path)
    print(f"\nSaved confusion matrix to: {csv_path}")


if __name__ == "__main__":
    main()
