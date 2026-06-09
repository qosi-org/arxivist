"""
evaluation/metrics.py — Trading Performance Metrics.

Implements all metrics reported in the paper (Section 5):
  - Mean episodic log-return (mu)
  - Sharpe ratio (S)
  - Cumulative out-performance over baseline (percentage points)
  - Turnover per episode
  - 95% bootstrapped confidence intervals

Paper: arXiv:2301.08688 — Section 5, Figure 2, Figure 3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mean_log_return(episode_returns: list[float]) -> float:
    """Mean episodic log-return across all evaluation episodes.

    Args:
        episode_returns: List of cumulative log-returns per episode.

    Returns:
        Mean log-return (mu in paper notation).
    """
    return float(np.mean(episode_returns))


def sharpe_ratio(episode_returns: list[float], risk_free: float = 0.0) -> float:
    """Sharpe ratio of episodic log-returns.

    Computed as (mean - risk_free) / std, matching the S notation in paper.

    Args:
        episode_returns: List of cumulative log-returns per episode.
        risk_free: Risk-free rate (default 0.0 as common for short-horizon HFT).

    Returns:
        Sharpe ratio (S in paper notation). Returns 0.0 if std is 0.
    """
    arr = np.array(episode_returns)
    std = float(np.std(arr, ddof=1))
    if std < 1e-10:
        return 0.0
    return float((np.mean(arr) - risk_free) / std)


def outperformance_pp(
    rl_returns: list[float], baseline_returns: list[float]
) -> float:
    """Cumulative outperformance of RL over baseline in percentage points.

    Paper reports this over 31 test episodes (Section 5).

    Args:
        rl_returns: Per-episode returns for RL strategy.
        baseline_returns: Per-episode returns for baseline strategy.

    Returns:
        Outperformance in percentage points (pp).
    """
    rl_cum = float(np.sum(rl_returns))
    base_cum = float(np.sum(baseline_returns))
    return (rl_cum - base_cum) * 100.0


def turnover(actions_per_episode: list[list[int]], skip_action: int = 6) -> float:
    """Mean number of non-skip actions per episode (turnover proxy).

    Args:
        actions_per_episode: List of action sequences, one per episode.
        skip_action: Index of the skip action (default 6).

    Returns:
        Mean turnover across episodes.
    """
    counts = [sum(1 for a in ep if a != skip_action) for ep in actions_per_episode]
    return float(np.mean(counts))


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Matches the 95% bootstrapped confidence intervals reported in Section 5.

    Args:
        values: Sample values (e.g. per-episode log-returns).
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples (default 10,000).
        seed: Random seed.

    Returns:
        (lower, upper) confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def print_results_table(
    results: dict[str, dict[str, float]],
) -> None:
    """Print a results table matching Table style in Figure 2 / Figure 3.

    Args:
        results: Dict mapping strategy name → dict of metric values.
            Expected keys: 'mean_return', 'sharpe', 'turnover'.
    """
    print(f"\n{'Strategy':<20} {'μ (mean return)':>18} {'Sharpe (S)':>12} {'Turnover':>10}")
    print("-" * 65)
    for name, vals in results.items():
        print(
            f"{name:<20} "
            f"{vals.get('mean_return', float('nan')):>18.4f} "
            f"{vals.get('sharpe', float('nan')):>12.2f} "
            f"{vals.get('turnover', float('nan')):>10.1f}"
        )
    print()
