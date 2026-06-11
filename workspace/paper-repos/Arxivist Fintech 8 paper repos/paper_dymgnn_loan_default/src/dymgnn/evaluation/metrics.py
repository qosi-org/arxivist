"""
evaluation/metrics.py — Evaluation Metrics for DYMGNN.

Implements the evaluation metrics reported in the paper (Section 4.3, 5):
  - AUC (Area Under ROC Curve)
  - F1 score (macro/binary)
  - 95% bootstrapped confidence intervals (Section 4.3)
  - SHAP feature importance (Section 5.4)

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional


def compute_auc(y_true: NDArray, y_score: NDArray) -> float:
    """Compute AUC-ROC (Section 4.3 primary metric).

    Args:
        y_true:  Ground truth binary labels [n].
        y_score: Predicted probabilities [n].

    Returns:
        AUC value in [0, 1].
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def compute_f1(y_true: NDArray, y_score: NDArray, threshold: float = 0.5) -> float:
    """Compute F1 score at a given threshold (Section 4.3 secondary metric).

    Args:
        y_true:     Ground truth binary labels [n].
        y_score:    Predicted probabilities [n].
        threshold:  Decision threshold (default 0.5).

    Returns:
        F1 score.
    """
    from sklearn.metrics import f1_score
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def bootstrap_ci(
    y_true: NDArray,
    y_score: NDArray,
    metric_fn,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrapped confidence interval for a metric (Section 4.3).

    Matches the CI computation described in the paper.

    Args:
        y_true:     Ground truth labels [n].
        y_score:    Predicted probabilities [n].
        metric_fn:  Function (y_true, y_score) → float.
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        seed:       Random seed.

    Returns:
        (lower, upper) confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_metrics = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bt, bs = y_true[idx], y_score[idx]
        if len(np.unique(bt)) < 2:
            continue  # skip degenerate bootstrap samples
        boot_metrics.append(metric_fn(bt, bs))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_metrics, 100 * alpha / 2))
    upper = float(np.percentile(boot_metrics, 100 * (1 - alpha / 2)))
    return lower, upper


def evaluate_full(
    y_true: NDArray,
    y_score: NDArray,
    threshold: float = 0.5,
    do_bootstrap: bool = True,
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    """Compute all reported metrics with 95% bootstrapped CIs.

    Matches the evaluation protocol described in Section 4.3 and Tables 3–7.

    Args:
        y_true:         Ground truth labels [n].
        y_score:        Predicted probabilities [n].
        threshold:      Binary decision threshold.
        do_bootstrap:   Whether to compute CI (slower but matches paper).
        n_bootstrap:    Number of bootstrap resamples.

    Returns:
        Dict with keys: auc, f1, auc_ci_lo, auc_ci_hi, f1_ci_lo, f1_ci_hi.
    """
    auc = compute_auc(y_true, y_score)
    f1 = compute_f1(y_true, y_score, threshold)

    results = {"auc": auc, "f1": f1}

    if do_bootstrap:
        auc_lo, auc_hi = bootstrap_ci(y_true, y_score, compute_auc, n_bootstrap=n_bootstrap)
        f1_lo, f1_hi = bootstrap_ci(
            y_true, y_score,
            lambda yt, ys: compute_f1(yt, ys, threshold),
            n_bootstrap=n_bootstrap,
        )
        results.update({
            "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
            "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
        })

    return results


def print_results_table(
    results: dict[str, dict],
    paper_results: Optional[dict] = None,
) -> None:
    """Print evaluation results table matching Tables 3–7 style.

    Args:
        results: Dict mapping model name → metrics dict.
        paper_results: Optional paper reported values for comparison.
    """
    header = f"{'Model':<25} {'AUC':>10} {'F1':>10}"
    print(f"\n{header}")
    print("-" * 50)
    for name, r in results.items():
        auc_str = f"{r['auc']:.3f}"
        f1_str = f"{r['f1']:.3f}"
        if "auc_ci_lo" in r:
            auc_str += f" ±{(r['auc_ci_hi']-r['auc_ci_lo'])/2:.3f}"
        if "f1_ci_lo" in r:
            f1_str += f" ±{(r['f1_ci_hi']-r['f1_ci_lo'])/2:.3f}"
        print(f"{name:<25} {auc_str:>10} {f1_str:>10}")

    if paper_results:
        print(f"\n  Paper reported (GAT-LSTM-ATT double):")
        print(f"    AUC = 0.812 ± 0.008 | F1 = 0.851 ± 0.007")
