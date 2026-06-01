#!/usr/bin/env python3
"""
train_qke.py — Quantum Kernel Estimator training and evaluation.

Reproduces Fig. 3b (decision boundary), Fig. 4 (kernel matrix), and QKE
classification success rates reported in the paper.

Usage:
    python scripts/train_qke.py --config configs/default.yaml
    python scripts/train_qke.py --dataset-id I --plot
    python scripts/train_qke.py --dry-run
    python scripts/train_qke.py --debug     # fast mode

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from qiskit_aer import AerSimulator

from qsvm.config import load_config, set_seed
from qsvm.data import SyntheticQuantumDataset
from qsvm.feature_map import FeatureMap
from qsvm.kernel_svm import QuantumKernelSVM
from qsvm.metrics import ClassificationMetrics
from qsvm.quantum_kernel import QuantumKernelEstimator
from qsvm.utils import (
    ensure_dir,
    plot_decision_boundary,
    plot_kernel_matrix,
    plot_success_vs_depth,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Quantum Kernel SVM")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset-id", choices=["I", "II", "III", None], default=None,
                   help="Run a single dataset; if None runs all three")
    p.add_argument("--use-statevector", action="store_true", default=True,
                   help="Use exact statevector kernel (default)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Fast: 5 data per label, 2 test sets")
    return p.parse_args()


# Dataset seeds matching the 3 paper datasets (arbitrary seeded splits)
DATASET_SEEDS = {"I": 42, "II": 137, "III": 271}


def run_dataset(
    dataset_id: str,
    cfg,
    feature_map: FeatureMap,
    backend: AerSimulator,
    out_dir: Path,
    plot: bool,
    debug: bool,
) -> dict:
    """Run QKE for one dataset. Returns {dataset_id: [success_rates]}."""
    seed = DATASET_SEEDS[dataset_id]
    n_per_label = 5 if debug else cfg.data.n_per_label
    n_test_sets = 2 if debug else cfg.qke.n_test_sets

    print(f"\n--- Dataset {dataset_id} (seed={seed}) ---")

    # Build training dataset
    dataset = SyntheticQuantumDataset(
        n_per_label=n_per_label,
        gap=cfg.data.gap,
        seed=seed,
        n_qubits=cfg.feature_map.n_qubits,
        domain_min=cfg.data.domain_min,
        domain_max=cfg.data.domain_max,
    )
    X_train, y_train = dataset.generate()
    print(f"  Train set: {len(X_train)} points "
          f"(+1: {(y_train==1).sum()}, -1: {(y_train==-1).sum()})")

    # Build quantum kernel estimator
    shots = 64 if debug else cfg.qke.shots_per_entry
    kernel = QuantumKernelEstimator(
        feature_map=feature_map,
        backend=backend,
        shots=shots,
        use_statevector=cfg.qke.use_statevector,
    )

    # Build and train QKE SVM
    svm = QuantumKernelSVM(kernel_estimator=kernel, C=cfg.qke.svm.C)
    svm.fit(X_train, y_train, verbose=cfg.output.verbose)

    # Save kernel matrix
    K_train = svm._K_train
    if cfg.output.save_kernel_matrix:
        np.save(out_dir / f"K_train_{dataset_id}.npy", K_train)

    # Print support vectors (matches Table S2 format)
    svs, alphas, sv_labels = svm.get_support_vectors()
    print(f"  Support vectors: {len(svs)}")
    b = svm.get_bias()
    print(f"  Bias: {b:.4f}")
    sv_table = {
        "dataset": dataset_id,
        "support_vectors": svs.tolist(),
        "alphas": alphas.tolist(),
        "labels": sv_labels.tolist(),
        "bias": b,
    }
    with open(out_dir / f"support_vectors_{dataset_id}.json", "w") as f:
        json.dump(sv_table, f, indent=2)

    # Evaluate on multiple test sets
    test_successes = []
    for ts_idx in range(n_test_sets):
        test_seed = seed + 500 + ts_idx * 17
        test_ds = SyntheticQuantumDataset(
            n_per_label=n_per_label,
            gap=cfg.data.gap,
            seed=test_seed,
            n_qubits=cfg.feature_map.n_qubits,
            domain_min=cfg.data.domain_min,
            domain_max=cfg.data.domain_max,
        )
        test_ds._V = dataset._V
        test_ds._VdagZ1Z2V = dataset._VdagZ1Z2V
        X_test, y_test = test_ds.generate()

        rate = svm.score(X_test, y_test, verbose=False)
        test_successes.append(rate)
        print(f"  Test set {ts_idx+1}: {rate*100:.1f}%")

    mean = np.mean(test_successes) * 100
    print(f"  Dataset {dataset_id} mean success: {mean:.2f}%  "
          f"(paper: Set {'I/II → 100%' if dataset_id != 'III' else 'III → 94.75%'})")

    # Plots
    if plot and cfg.output.save_plots:
        # Fig 4a — kernel matrix heatmap
        fig = plot_kernel_matrix(
            K_estimated=K_train,
            K_ideal=None,   # would need noiseless separate run
            title=f"Kernel matrix — Dataset {dataset_id}",
            cut_row=K_train.shape[0] // 2,
            output_path=out_dir / f"kernel_{dataset_id}.{cfg.output.plot_format}",
            dpi=cfg.output.plot_dpi,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Fig 3b — decision boundary (last test set)
        fig = plot_decision_boundary(
            qke_model=svm,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            resolution=50,
            domain=(cfg.data.domain_min, cfg.data.domain_max),
            output_path=out_dir / f"boundary_{dataset_id}.{cfg.output.plot_format}",
            dpi=cfg.output.plot_dpi,
        )
        plt.close(fig)
        print(f"  Plots saved → {out_dir}/")

    return {dataset_id: test_successes}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output.results_dir = args.output_dir

    set_seed(cfg.seed)

    out_dir = ensure_dir(Path(cfg.output.results_dir) / "qke")

    print("=" * 55)
    print("  Quantum Kernel Estimator (QKE)")
    print(f"  Paper: Havlicek et al. (2018), arXiv:1804.11326v2")
    print("=" * 55)
    mode = "exact statevector" if cfg.qke.use_statevector else f"shot-based ({cfg.qke.shots_per_entry} shots)"
    print(f"  Kernel mode  : {mode}")
    print(f"  SVM C        : {cfg.qke.svm.C}")
    print(f"  Output       : {out_dir}")
    print("=" * 55)

    if args.dry_run:
        print("\n[DRY-RUN] All components loaded. Exiting.")
        return

    feature_map = FeatureMap(
        n_qubits=cfg.feature_map.n_qubits,
        reps=cfg.feature_map.reps,
    )
    backend = AerSimulator(method="statevector" if cfg.qke.use_statevector else "automatic")

    datasets = [args.dataset_id] if args.dataset_id else ["I", "II", "III"]
    all_qke_results = {}

    t_start = time.time()
    for ds_id in datasets:
        result = run_dataset(
            ds_id, cfg, feature_map, backend, out_dir,
            plot=args.plot, debug=args.debug,
        )
        all_qke_results.update(result)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Summary
    ClassificationMetrics.print_summary({}, all_qke_results)

    # Save aggregated results
    results_file = out_dir / "qke_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "paper": "Havlicek et al. 2018",
            "protocol": "QKE",
            "kernel_mode": "statevector" if cfg.qke.use_statevector else "shots",
            "results": {k: v for k, v in all_qke_results.items()},
            "means": {k: float(np.mean(v)) for k, v in all_qke_results.items()},
        }, f, indent=2)
    print(f"Results saved → {results_file}")
    print("Done.")


if __name__ == "__main__":
    main()
