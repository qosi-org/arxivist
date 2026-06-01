#!/usr/bin/env python3
"""
train_qvc.py — Quantum Variational Classifier training and evaluation.

Reproduces Fig. 3a (cost convergence) and Fig. 3c (QVC success rates).

Usage:
    python scripts/train_qvc.py --config configs/default.yaml
    python scripts/train_qvc.py --depth 4 --n-datasets 1 --seed 7
    python scripts/train_qvc.py --dry-run      # validates setup only
    python scripts/train_qvc.py --debug        # fast run with n_iter=5

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from qiskit_aer import AerSimulator

from qsvm.config import Config, load_config, set_seed
from qsvm.data import SyntheticQuantumDataset
from qsvm.feature_map import FeatureMap
from qsvm.metrics import ClassificationMetrics
from qsvm.utils import (
    ensure_dir,
    plot_cost_convergence,
    plot_success_vs_depth,
)
from qsvm.variational_classifier import QuantumVariationalClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Train Quantum Variational Classifier")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    p.add_argument("--depth", type=int, default=None,
                   help="Single depth to run (overrides config.qvc.depths)")
    p.add_argument("--n-datasets", type=int, default=None,
                   help="Number of random training datasets (overrides config)")
    p.add_argument("--seed", type=int, default=None, help="Random seed override")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (overrides config)")
    p.add_argument("--plot", action="store_true", help="Save plots")
    p.add_argument("--dry-run", action="store_true",
                   help="Build all components, skip training")
    p.add_argument("--debug", action="store_true",
                   help="Fast mode: n_iter=5, 1 dataset, depths=[0,4]")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    if args.seed is not None:
        cfg.seed = args.seed
    if args.depth is not None:
        cfg.qvc.depths = [args.depth]
    if args.n_datasets is not None:
        cfg.qvc.n_datasets = args.n_datasets
    if args.output_dir is not None:
        cfg.output.results_dir = args.output_dir
    if args.debug:
        cfg.qvc.spsa.n_iter = 5
        cfg.qvc.n_datasets = 1
        cfg.qvc.depths = [0, 4]
        cfg.qvc.spsa.shots_cost = 32
        cfg.qvc.spsa.shots_classify = 100
        print("[DEBUG] Fast mode: n_iter=5, 1 dataset, depths=[0,4]")

    set_seed(cfg.seed)

    out_dir = ensure_dir(Path(cfg.output.results_dir) / "qvc")

    # Print summary
    print("=" * 55)
    print("  Quantum Variational Classifier (QVC)")
    print(f"  Paper: Havlicek et al. (2018), arXiv:1804.11326v2")
    print("=" * 55)
    print(f"  Config     : {args.config}")
    print(f"  Seed       : {cfg.seed}")
    print(f"  Depths     : {cfg.qvc.depths}")
    print(f"  Datasets   : {cfg.qvc.n_datasets} per depth")
    print(f"  SPSA iters : {cfg.qvc.spsa.n_iter}")
    print(f"  Shots cost : {cfg.qvc.spsa.shots_cost}")
    print(f"  Backend    : {cfg.backend.name}")
    print(f"  Output     : {out_dir}")
    print("=" * 55)

    if args.dry_run:
        print("\n[DRY-RUN] All components loaded successfully. Exiting.")
        return

    # Build shared components
    feature_map = FeatureMap(
        n_qubits=cfg.feature_map.n_qubits,
        reps=cfg.feature_map.reps,
    )
    backend = AerSimulator(method="automatic")

    # Results accumulator: {depth: [success_rate_1, ...]}
    all_results: dict = {}
    cost_histories: dict = {}   # {depth: [cost_list_1, ...]}

    t_start = time.time()

    for depth in cfg.qvc.depths:
        print(f"\n--- Depth l={depth} ---")
        all_results[depth] = []
        cost_histories[depth] = []

        for ds_idx in range(cfg.qvc.n_datasets):
            ds_seed = cfg.seed + ds_idx * 31 + depth * 7
            print(f"  Dataset {ds_idx+1}/{cfg.qvc.n_datasets} (seed={ds_seed})")

            # Generate training data
            dataset = SyntheticQuantumDataset(
                n_per_label=cfg.data.n_per_label,
                gap=cfg.data.gap,
                seed=ds_seed,
                n_qubits=cfg.feature_map.n_qubits,
                domain_min=cfg.data.domain_min,
                domain_max=cfg.data.domain_max,
            )
            X_train, y_train = dataset.generate()
            print(f"    Train set: {len(X_train)} points "
                  f"(+1: {(y_train==1).sum()}, -1: {(y_train==-1).sum()})")

            # Build and train QVC
            qvc = QuantumVariationalClassifier(
                feature_map=feature_map,
                depth=depth,
                backend=backend,
                shots=cfg.qvc.spsa.shots_eval,
            )

            theta_star, b_star, cost_hist = qvc.fit(
                X_train, y_train,
                n_iter=cfg.qvc.spsa.n_iter,
                shots_cost=cfg.qvc.spsa.shots_cost,
                a=cfg.qvc.spsa.a,
                c=cfg.qvc.spsa.c,
                A=cfg.qvc.spsa.A,
                alpha_spsa=cfg.qvc.spsa.alpha_spsa,
                gamma_spsa=cfg.qvc.spsa.gamma_spsa,
                bias_range=tuple(cfg.qvc.bias_range),
                verbose=cfg.output.verbose,
            )
            cost_histories[depth].append(cost_hist)
            print(f"    Final R_emp: {cost_hist[-1]:.4f}  |  b*={b_star:.4f}")

            # Classify multiple test sets  [paper: 20 test sets per dataset]
            n_test_sets = cfg.qvc.spsa.n_iter  # use config n_test_sets
            n_test_sets = getattr(cfg.data, 'n_test_sets', 10)
            test_successes = []
            for ts_idx in range(n_test_sets):
                test_seed = ds_seed + 500 + ts_idx * 13
                test_ds = SyntheticQuantumDataset(
                    n_per_label=cfg.data.n_per_label,
                    gap=cfg.data.gap,
                    seed=test_seed,
                    n_qubits=cfg.feature_map.n_qubits,
                    domain_min=cfg.data.domain_min,
                    domain_max=cfg.data.domain_max,
                )
                # Use same V as training set
                test_ds._V = dataset._V
                test_ds._VdagZ1Z2V = dataset._VdagZ1Z2V
                X_test, y_test = test_ds.generate()

                rate = qvc.score(
                    X_test, y_test, theta_star,
                    b=0.0,   # b*=0 at inference — paper
                    shots=cfg.qvc.spsa.shots_classify,
                )
                test_successes.append(rate)

            mean_rate = np.mean(test_successes)
            all_results[depth].extend(test_successes)
            print(f"    Test success: {mean_rate*100:.1f}% "
                  f"(over {n_test_sets} test sets)")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Summary
    ClassificationMetrics.print_summary(all_results, {})

    # Save results
    results_file = out_dir / "qvc_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "paper": "Havlicek et al. 2018",
            "protocol": "QVC",
            "config_seed": cfg.seed,
            "results": {str(d): v for d, v in all_results.items()},
            "aggregated": ClassificationMetrics.depth_vs_accuracy(all_results),
        }, f, indent=2)
    print(f"Results saved → {results_file}")

    # Plots
    if args.plot and cfg.output.save_plots:
        fig = plot_cost_convergence(
            cost_histories,
            output_path=out_dir / f"cost_convergence.{cfg.output.plot_format}",
            dpi=cfg.output.plot_dpi,
        )
        plt.close(fig)
        fig = plot_success_vs_depth(
            all_results,
            qke_results={},
            output_path=out_dir / f"success_vs_depth.{cfg.output.plot_format}",
            dpi=cfg.output.plot_dpi,
        )
        plt.close(fig)
        print(f"Plots saved → {out_dir}/")

    print("Done.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
