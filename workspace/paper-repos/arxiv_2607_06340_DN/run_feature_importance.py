#!/usr/bin/env python
"""
run_feature_importance.py — reproduce Figure 6.4: built-in vs permutation
feature importance for a trained experiment.

Usage:
    python run_feature_importance.py --results-dir results/6.2 --out results/importance/
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sig_vol_id.evaluation.importance import ImportanceAnalyzer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare built-in vs permutation feature importance")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--out", type=str, default="results/importance/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = Path(args.results_dir) / "model_and_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} not found. Run train.py first.")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    clf, X_test, y_test = data["clf"], data["X_test"], data["y_test"]

    analyzer = ImportanceAnalyzer()
    builtin = analyzer.builtin_importance(clf)
    perm = analyzer.permutation_importance(clf, X_test, y_test)

    print("Top 10 by built-in (gain) importance:")
    print(builtin.head(10).to_string())
    print("\nTop 10 by permutation importance:")
    print(perm.head(10).to_string())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    builtin.to_csv(out_dir / "builtin_importance.csv")
    perm.to_csv(out_dir / "permutation_importance.csv")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        analyzer.truncate_at_cumulative(builtin, 0.9).plot.barh(ax=axes[0], title="XGBoost Built-in Importance")
        analyzer.truncate_at_cumulative(perm, 0.9).plot.barh(ax=axes[1], title="Permutation Importance")
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()
        plt.tight_layout()
        fig_path = out_dir / "figure6_4_reproduction.png"
        plt.savefig(fig_path, dpi=150)
        print(f"\nSaved plot to: {fig_path}")
    except ImportError:
        print("matplotlib not available; skipped plot.")


if __name__ == "__main__":
    main()
