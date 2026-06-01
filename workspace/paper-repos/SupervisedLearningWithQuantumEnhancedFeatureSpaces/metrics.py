"""
metrics.py — Classification Metrics

Evaluation utilities for reproducing Tables and Figures from
Havlicek et al. (2018).

Primary metric: Classification success rate = fraction of correctly labelled
test points, as reported in Fig. 3c.

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

from __future__ import annotations

from typing import Dict, List, Any

import numpy as np


class ClassificationMetrics:
    """
    Computes and aggregates classification metrics for QVC and QKE protocols.
    """

    @staticmethod
    def success_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Classification success rate = fraction of correct predictions.

        As reported in Fig. 3c: "Classification success [100%]".

        Parameters
        ----------
        y_true : np.ndarray, shape [N], values in {+1, -1}
        y_pred : np.ndarray, shape [N], values in {+1, -1}

        Returns
        -------
        float in [0, 1]
        """
        assert y_true.shape == y_pred.shape, (
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def depth_vs_accuracy(
        results: Dict[int, List[float]],
    ) -> Dict[str, Any]:
        """
        Aggregate QVC success rates by circuit depth for Fig. 3c.

        Parameters
        ----------
        results : Dict[int, List[float]]
            {depth: [success_rate_set1, ..., success_rate_setN]}
            Each entry is a list of success rates across N datasets × test sets.

        Returns
        -------
        Dict with keys: depths, means, stds, stderr
        """
        depths = sorted(results.keys())
        means, stds, stderrs = [], [], []

        for d in depths:
            vals = np.array(results[d])
            means.append(float(vals.mean()))
            stds.append(float(vals.std()))
            # Standard error of the mean (paper uses this for error bars in Fig. 3c)
            stderrs.append(float(vals.std() / np.sqrt(len(vals))))

        return {
            "depths": depths,
            "means": means,
            "stds": stds,
            "stderr": stderrs,
        }

    @staticmethod
    def summarise_qke_results(
        results: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Aggregate QKE results per dataset (Set I, II, III).

        Parameters
        ----------
        results : Dict[str, List[float]]
            {"Set I": [rate_1, ..., rate_10], "Set II": [...], "Set III": [...]}

        Returns
        -------
        Dict with per-set mean success rates.
        """
        summary = {}
        for name, rates in results.items():
            arr = np.array(rates)
            summary[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "n_test_sets": len(rates),
            }
        return summary

    @staticmethod
    def print_summary(
        qvc_depth_results: Dict[int, List[float]],
        qke_set_results: Dict[str, List[float]],
    ) -> None:
        """Print a formatted summary table matching paper's reported results."""
        print("\n" + "=" * 55)
        print("  REPRODUCTION RESULTS — Havlicek et al. (2018)")
        print("=" * 55)

        # QVC
        print("\n  Quantum Variational Classifier (QVC)")
        print("  " + "-" * 40)
        print(f"  {'Depth':<8} {'Mean Success':>14} {'Std Dev':>10}")
        agg = ClassificationMetrics.depth_vs_accuracy(qvc_depth_results)
        for d, m, s in zip(agg["depths"], agg["means"], agg["stds"]):
            marker = " ← paper: ~100%" if d >= 1 else " ← paper: ~60-75%"
            print(f"  {d:<8} {m*100:>13.1f}% {s*100:>9.1f}%{marker}")

        # QKE
        print("\n  Quantum Kernel Estimator (QKE)")
        print("  " + "-" * 40)
        paper_qke = {"Set I": 100.0, "Set II": 100.0, "Set III": 94.75}
        for name, rates in qke_set_results.items():
            mean = np.mean(rates) * 100
            paper = paper_qke.get(name, "?")
            print(f"  {name}: {mean:.2f}%   (paper: {paper}%)")

        print("\n" + "=" * 55 + "\n")
