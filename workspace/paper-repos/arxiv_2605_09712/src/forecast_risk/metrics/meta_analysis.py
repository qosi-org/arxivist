"""
forecast_risk.metrics.meta_analysis
=====================================
Cross-sectional meta-analysis metrics aggregated over a design space
of targets, horizons, and evaluation samples.

Implements Section 2.5 of:
  "Quantifying the Risk-Return Tradeoff in Forecasting"
  Philippe Goulet Coulombe, arXiv: 2605.09712

The "return" is defined as percentage improvement over the benchmark:
  R^M_{v,h,s} = (P^B_{v,h,s} - P^M_{v,h,s}) / P^B_{v,h,s} * 100

Meta-Sharpe, Meta-Sortino, Meta-Omega, and Meta-Edge are then computed
over the cross-sectional distribution {R^M_i} for i = 1,...,N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .risk_metrics import ForecastRiskMetrics
from .edge_ratio import EdgeRatioCalculator


def percentage_return(
    perf_model: np.ndarray, perf_benchmark: np.ndarray
) -> np.ndarray:
    """
    Compute percentage return relative to benchmark.

    Paper: Sec 2.5, Eq: R^M_{v,h,s} = (P^B - P^M) / P^B * 100

    Args:
        perf_model:     Model performance metric (lower = better, e.g. RMSE) [N].
        perf_benchmark: Benchmark performance metric [N].

    Returns:
        Percentage improvement [N]. Positive = model beats benchmark.
    """
    assert perf_model.shape == perf_benchmark.shape
    # Guard against zero benchmark
    denom = np.where(np.abs(perf_benchmark) > 1e-12, perf_benchmark, 1e-12)
    return (perf_benchmark - perf_model) / np.abs(denom) * 100.0


class MetaAnalysisMetrics:
    """
    Meta-analysis metrics over a cross-sectional design space.

    Paper: Section 2.5 — Meta-Analysis Statistics
    "Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

    The same Sharpe/Sortino/Omega/Edge logic from time-series evaluation is
    applied over the distribution of {R^M_i} across the design space, rather
    than over time.
    """

    def __init__(self, eps: float = 1e-10):
        self._rm = ForecastRiskMetrics(eps=eps)
        self._er = EdgeRatioCalculator(eps=eps)
        self.eps = eps

    def __repr__(self) -> str:
        return f"MetaAnalysisMetrics(eps={self.eps})"

    def meta_sharpe(self, returns_flat: np.ndarray) -> float:
        """
        Meta-Sharpe = R_bar^M / s^M_R  (cross-sectional Sharpe).

        Args:
            returns_flat: Percentage returns flattened over design space [N].
        """
        return self._rm.sharpe(returns_flat)

    def meta_sortino(self, returns_flat: np.ndarray) -> float:
        """Meta-Sortino (cross-sectional Sortino)."""
        return self._rm.sortino(returns_flat)

    def meta_omega(self, returns_flat: np.ndarray) -> float:
        """Meta-Omega (cross-sectional Omega)."""
        return self._rm.omega(returns_flat)

    def full_table(
        self,
        perf_dict: dict[str, np.ndarray],
        benchmark_key: str,
        model_keys: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate meta-analysis table for all models (mirrors Table 1/2 in paper).

        Args:
            perf_dict:     {model_name: performance_array [N]} over design space.
            benchmark_key: Key of the benchmark model in perf_dict.
            model_keys:    Subset of models to include (None = all except benchmark).

        Returns:
            DataFrame with columns [Return, Vol, Sharpe, Sortino, Omega, Edge].
        """
        bench = perf_dict[benchmark_key]

        if model_keys is None:
            model_keys = [k for k in perf_dict if k != benchmark_key]

        # Build percentage-return matrix [M, N] for Edge Ratio computation
        ret_matrix = np.stack(
            [percentage_return(perf_dict[k], bench) for k in model_keys], axis=0
        )  # [M, N]

        rows = []
        for i, key in enumerate(model_keys):
            r = ret_matrix[i]  # [N]
            edge = self._er.compute(ret_matrix, i)

            rows.append({
                "Model": key,
                "Return": float(np.mean(r)),
                "Vol": float(np.std(r, ddof=1)),
                "Sharpe": self.meta_sharpe(r),
                "Sortino": self.meta_sortino(r),
                "Omega": self.meta_omega(r),
                "Edge": edge,
            })

        df = pd.DataFrame(rows).set_index("Model")
        return df
