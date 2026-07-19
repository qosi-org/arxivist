#!/usr/bin/env python
"""
inference.py — classify a small fresh batch of paths using a saved model.

Usage:
    python inference.py --config configs/config.yaml --experiment 6.1 --n-paths 100
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sig_vol_id.data.experiment_builder import EXPERIMENTS, ExperimentBuilder  # noqa: E402
from sig_vol_id.utils.config import Config, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify fresh paths with a saved model")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--experiment", type=str, required=True, choices=list(EXPERIMENTS))
    p.add_argument("--n-paths", type=int, default=100)
    p.add_argument("--seed", type=int, default=999)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    set_global_seed(args.seed)

    pkl_path = Path("results") / args.experiment / "model_and_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} not found. Run train.py --experiment {args.experiment} first.")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    clf, class_names = data["clf"], data["class_names"]

    builder = ExperimentBuilder(cfg)
    _, _, X_fresh, y_fresh, _ = builder.build(args.experiment, 1, args.n_paths, args.seed)

    acc = clf.accuracy(X_fresh, y_fresh)
    print(f"Fresh-batch accuracy on {args.n_paths} paths/class: {acc:.4f}")
    preds = clf.predict(X_fresh)
    for i in range(min(10, len(preds))):
        print(f"  true={class_names[y_fresh[i]]:>8s}  predicted={class_names[preds[i]]:>8s}")


if __name__ == "__main__":
    main()
