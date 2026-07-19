"""
causal/estimator.py
--------------------
Implements EQ2 (Percentage Lift Estimator) and bootstrap confidence intervals.

Paper: Kamat (2026), Section 6.3 — Naive random-matched results.

EQ2:
    Lift(%) = (Ȳ_treated - Ȳ_control) / Ȳ_control × 100

Bootstrap: 1,000 iterations, percentile method (2.5th and 97.5th percentiles).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from red_cohort.utils.config import CausalConfig


class LiftEstimator:
    """
    Estimates the percentage lift in an outcome variable between treated
    and control groups, with bootstrap confidence intervals.

    Paper reference:
        Section 6.3 — Table 4. Key results:
          - first_30min_buyer_count: +132.3% [+127.0%, +137.4%]
          - first_30min_sol_inflow:  +136.5% [+120.9%, +152.2%]

    Args:
        config: CausalConfig with bootstrap_iterations, bootstrap_ci_level, random_seed.
    """

    def __init__(self, config: CausalConfig) -> None:
        self.cfg = config

    def compute_lift(
        self,
        treated_outcomes: pd.Series,
        control_outcomes: pd.Series,
    ) -> float:
        """
        EQ2: Lift(%) = (mean_treated - mean_control) / mean_control × 100

        Args:
            treated_outcomes: Outcome values for the treated group.
            control_outcomes: Outcome values for the control group.

        Returns:
            Point estimate of percentage lift.
        """
        mean_treated = treated_outcomes.mean()
        mean_control = control_outcomes.mean()

        if mean_control == 0:
            raise ValueError("Control mean is zero — cannot compute lift.")

        # EQ2 (Section 6.3)
        return (mean_treated - mean_control) / mean_control * 100.0

    def bootstrap_ci(
        self,
        treated_outcomes: pd.Series,
        control_outcomes: pd.Series,
        n_iter: Optional[int] = None,
        ci_level: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap percentile confidence interval for the lift estimate.

        Method: resample treated and control independently with replacement,
        compute lift on each resample, return [alpha/2, 1-alpha/2] percentiles.

        Paper: "bootstrap 95% confidence intervals (1,000 iterations, percentile method)"
               — Section 6.3.

        Args:
            treated_outcomes, control_outcomes: Outcome Series.
            n_iter: Bootstrap iterations (default from config: 1,000).
            ci_level: CI level (default from config: 0.95).
            seed: Random seed (default from config: 42).

        Returns:
            (ci_lower, ci_upper) tuple.
        """
        n_iter = n_iter or self.cfg.bootstrap_iterations
        ci_level = ci_level or self.cfg.bootstrap_ci_level
        seed = seed or self.cfg.random_seed

        rng = np.random.default_rng(seed)
        treated_arr = treated_outcomes.to_numpy()
        control_arr = control_outcomes.to_numpy()

        boot_lifts = np.empty(n_iter)
        for i in range(n_iter):
            t_sample = rng.choice(treated_arr, size=len(treated_arr), replace=True)
            c_sample = rng.choice(control_arr, size=len(control_arr), replace=True)
            c_mean = c_sample.mean()
            if c_mean == 0:
                boot_lifts[i] = 0.0
            else:
                boot_lifts[i] = (t_sample.mean() - c_mean) / c_mean * 100.0

        alpha = 1.0 - ci_level
        ci_lower = float(np.percentile(boot_lifts, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_lifts, 100 * (1 - alpha / 2)))
        return ci_lower, ci_upper

    def estimate(
        self,
        treated_df: pd.DataFrame,
        control_df: pd.DataFrame,
        outcome_col: str,
    ) -> Dict:
        """
        Full estimation: point lift + bootstrap CI for one outcome variable.

        Args:
            treated_df: Treated group outcomes DataFrame.
            control_df: Control group outcomes DataFrame.
            outcome_col: Column name to evaluate (e.g., 'first_30min_buyer_count').

        Returns:
            dict with: point_estimate, ci_lower, ci_upper, treated_mean, control_mean,
                       n_treated, n_control, outcome_col.
        """
        treated_outcomes = treated_df[outcome_col].dropna()
        control_outcomes = control_df[outcome_col].dropna()

        point_lift = self.compute_lift(treated_outcomes, control_outcomes)
        ci_lower, ci_upper = self.bootstrap_ci(treated_outcomes, control_outcomes)

        return {
            "outcome_col": outcome_col,
            "treated_mean": round(treated_outcomes.mean(), 4),
            "control_mean": round(control_outcomes.mean(), 4),
            "point_estimate_pct": round(point_lift, 2),
            "ci_lower_pct": round(ci_lower, 2),
            "ci_upper_pct": round(ci_upper, 2),
            "n_treated": len(treated_outcomes),
            "n_control": len(control_outcomes),
        }

    def __repr__(self) -> str:
        return (
            f"LiftEstimator(n_iter={self.cfg.bootstrap_iterations}, "
            f"ci={self.cfg.bootstrap_ci_level}, seed={self.cfg.random_seed})"
        )
