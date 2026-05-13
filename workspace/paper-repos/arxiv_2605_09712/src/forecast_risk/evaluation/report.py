"""
forecast_risk.evaluation.report
=================================
Generate full risk-adjusted report tables, mirroring Tables 4-15 in the paper.

Paper: "Quantifying the Risk-Return Tradeoff in Forecasting"
Philippe Goulet Coulombe, arXiv: 2605.09712

Each table reports (for squared error and absolute error):
  Panel A/B: Return, Sharpe, Sortino, Omega, MaxDD, Edge
  Panel C:   RMSE ratio, MAE ratio, rho(1), DM t-stat
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from ..metrics.risk_metrics import ForecastRiskMetrics, compute_returns
from ..metrics.edge_ratio import EdgeRatioCalculator
from ..utils.dm_test import DieboldMarianoTest


class RiskAdjustedReport:
    """
    Generates a complete risk-adjusted performance table for a set of models.

    Paper: Tables 4-15 — full appendix results.
    "Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

    Args:
        horizon: Forecast horizon h (used for DM test HAC truncation).
        eps:     Division guard for metrics.
    """

    def __init__(self, horizon: int = 1, eps: float = 1e-10):
        self.horizon = horizon
        self._rm = ForecastRiskMetrics(eps=eps)
        self._er = EdgeRatioCalculator(eps=eps)
        self._dm = DieboldMarianoTest(h=horizon)

    def generate(
        self,
        losses_dict: dict[str, np.ndarray],
        benchmark_key: str,
        loss_fn: str = "squared_error",
    ) -> pd.DataFrame:
        """
        Generate full risk-adjusted metrics table.

        Args:
            losses_dict:   {model_name: loss_series [T]} for all models including benchmark.
            benchmark_key: Key of the benchmark model.
            loss_fn:       Name of loss function used (for labelling).

        Returns:
            DataFrame indexed by model name with columns:
            [Return, Sharpe, Sortino, Omega, MaxDD, Edge, RMSE_ratio, rho1, DM_tstat]
        """
        benchmark_losses = losses_dict[benchmark_key]
        model_keys = [k for k in losses_dict if k != benchmark_key]

        # Build [M, T] loss matrix for Edge Ratio
        all_keys = [benchmark_key] + model_keys
        T = len(benchmark_losses)
        loss_matrix = np.stack([losses_dict[k] for k in all_keys], axis=0)  # [M+1, T]

        rows = []
        for i, key in enumerate(model_keys):
            model_losses = losses_dict[key]
            # Return series: r_t = L^B_t - L^M_t
            r = compute_returns(benchmark_losses, model_losses)

            # Mask NaN periods
            valid = ~(np.isnan(r) | np.isinf(r))
            r_clean = r[valid]
            bl_clean = benchmark_losses[valid]
            ml_clean = model_losses[valid]

            if len(r_clean) < 3:
                rows.append({"Model": key, **{c: np.nan for c in _COLUMNS}})
                continue

            # Risk metrics
            ret_mean = float(np.mean(r_clean))
            sharpe = self._rm.sharpe(r_clean)
            sortino = self._rm.sortino(r_clean)
            omega = self._rm.omega(r_clean)
            max_dd = -self._rm.max_drawdown(r_clean)  # negative = bad

            # Edge ratio using model index in full loss matrix
            model_idx_in_matrix = i + 1  # index 0 is benchmark
            edge = self._er.compute(loss_matrix[:, valid], model_idx_in_matrix)

            # Classical metrics
            rmse_ratio = float(
                np.sqrt(np.mean(ml_clean)) / max(np.sqrt(np.mean(bl_clean)), 1e-12)
            )

            # First-order autocorrelation rho(1) — reported in paper tables
            if len(r_clean) > 2:
                rho1 = float(np.corrcoef(r_clean[:-1], r_clean[1:])[0, 1])
            else:
                rho1 = float("nan")

            # Diebold-Mariano t-statistic
            dm_stat = self._dm.statistic(bl_clean, ml_clean)

            rows.append({
                "Model": key,
                "Return": ret_mean,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Omega": omega,
                "MaxDD": max_dd,
                "Edge": edge,
                "RMSE_ratio": rmse_ratio,
                "rho1": rho1,
                "DM_tstat": dm_stat,
            })

        df = pd.DataFrame(rows).set_index("Model")
        return df

    def to_latex(self, df: pd.DataFrame, caption: str = "") -> str:
        """Export table to LaTeX format."""
        return df.round(2).to_latex(caption=caption, escape=False)

    def to_excel(self, df: pd.DataFrame, path: str) -> None:
        """Export table to Excel."""
        df.round(4).to_excel(path)


_COLUMNS = ["Return", "Sharpe", "Sortino", "Omega", "MaxDD", "Edge",
            "RMSE_ratio", "rho1", "DM_tstat"]
