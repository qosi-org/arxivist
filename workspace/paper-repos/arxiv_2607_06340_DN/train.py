#!/usr/bin/env python
"""
train.py — run one named experiment: simulate paths, compute signatures,
train XGBoost, save the model + test set + results.

Usage:
    python train.py --config configs/config.yaml --experiment 6.1
    python train.py --config configs/config.yaml --experiment 5.1 --debug
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sig_vol_id.data.experiment_builder import EXPERIMENTS, ExperimentBuilder  # noqa: E402
from sig_vol_id.models.xgb_classifier import SignatureXGBClassifier  # noqa: E402
from sig_vol_id.utils.config import Config, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a signature+XGBoost volatility-model classifier")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--experiment", type=str, required=True, choices=list(EXPERIMENTS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true", help="Tiny smoke-test scale")
    p.add_argument("--dry-run", action="store_true", help="Validate config/experiment only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.debug:
        cfg = cfg.apply_debug_overrides()
        print("[--debug] n_paths_per_class=500, n_test_per_class=200, n_steps=20")

    set_global_seed(args.seed)

    n_paths = cfg["simulation"]["n_paths_per_class"]
    n_test = cfg["simulation"]["n_test_per_class"]
    print(f"Experiment: {args.experiment}")
    print(f"Classes: {EXPERIMENTS[args.experiment]['classes']}")
    print(f"Paths/class: {n_paths} (+{n_test} test)")

    builder = ExperimentBuilder(cfg)

    if args.dry_run:
        print("[--dry-run] Experiment resolved successfully. Exiting without simulating.")
        return

    X_train, y_train, X_test, y_test, class_names = builder.build(
        args.experiment, n_paths, n_test, args.seed
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    clf = SignatureXGBClassifier(
        n_classes=len(class_names),
        learning_rate=cfg["xgboost"]["learning_rate"],
        max_depth=cfg["xgboost"]["max_depth"],
        n_estimators=cfg["xgboost"]["n_estimators"],
        tree_method=cfg["xgboost"]["tree_method"],
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.accuracy(X_train, y_train)
    test_acc = clf.accuracy(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    out_dir = Path("results") / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_and_data.pkl", "wb") as f:
        pickle.dump(
            {
                "clf": clf,
                "X_test": X_test,
                "y_test": y_test,
                "class_names": class_names,
                "train_acc": train_acc,
                "test_acc": test_acc,
            },
            f,
        )
    print(f"Saved to: {out_dir / 'model_and_data.pkl'}")
    print(f"Next: python evaluate.py --results-dir results/{args.experiment} --experiment {args.experiment}")


if __name__ == "__main__":
    main()
