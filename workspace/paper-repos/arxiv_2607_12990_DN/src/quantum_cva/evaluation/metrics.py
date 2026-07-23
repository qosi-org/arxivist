"""
Trajectory-level metrics: query-cost accounting and log-log query-scaling
exponent fits.

Implements Section 3.2.2 (Eq. 42) and Appendix B (median-error scaling
proposition) of arXiv:2607.12990.

SIR reference: architecture.modules "evaluation/metrics.py", mathematical_spec
"Ideal QAE additive error bound".
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class TrajectoryMetrics:
    """Query-cost accounting and query-scaling exponent fitting for adaptive
    amplitude-estimation trajectories."""

    def __repr__(self) -> str:  # noqa: D105
        return "TrajectoryMetrics()"

    def query_cost(self, stages: List[Tuple[int, int]]) -> int:
        """Accumulated oracle-query cost N_q(S) = sum_s n_s * (2*k_s + 1)  (Eq. 42).

        Args:
            stages: list of (k_s, n_s) pairs -- Grover power and shot count
                used at each adaptive stage s.

        Returns:
            Total accumulated query cost N_q.
        """
        return sum(n_s * (2 * k_s + 1) for k_s, n_s in stages)

    def fit_scaling_exponent(
        self, n_q_values: np.ndarray, errors: np.ndarray
    ) -> Tuple[float, float, float]:
        """Fit log(error) = alpha + beta * log(N_q) via least squares
        (Table 2). beta ~ -1 indicates amplified QAE-like O(1/N_q) scaling;
        beta ~ -0.5 indicates classical Monte-Carlo-like O(1/sqrt(N_q))
        scaling (Appendix B).

        Args:
            n_q_values: array of query-cost values (binned median N_q).
            errors: array of corresponding median relative errors.

        Returns:
            (beta, alpha, r_squared).
        """
        log_n = np.log(n_q_values)
        log_e = np.log(errors)
        beta, alpha = np.polyfit(log_n, log_e, 1)
        pred = alpha + beta * log_n
        ss_res = np.sum((log_e - pred) ** 2)
        ss_tot = np.sum((log_e - log_e.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return float(beta), float(alpha), float(r_squared)

    def median_with_bootstrap_ci(
        self, values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 100000
    ) -> Tuple[float, float, float]:
        """Bootstrap-resampled median and confidence interval (Section 3.2.2,
        footnote on bootstrap resampling for median-based trajectory
        summaries).

        Args:
            values: array of per-trajectory values (e.g. relative errors in
                one log-Nq bin).
            n_bootstrap: number of bootstrap resamples.
            ci: confidence level (e.g. 0.95 for a 95% interval).
            seed: RNG seed.

        Returns:
            (median, ci_lower, ci_upper).
        """
        rng = np.random.default_rng(seed)
        n = len(values)
        boot_medians = np.array(
            [np.median(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)]
        )
        alpha = 1 - ci
        lo = float(np.percentile(boot_medians, 100 * alpha / 2))
        hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
        return float(np.median(values)), lo, hi
