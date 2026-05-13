"""
forecast_risk.metrics.risk_metrics
===================================
Core risk-adjusted performance metrics for forecast evaluation.

Implements metrics from Section 2.2 of:
  "Quantifying the Risk-Return Tradeoff in Forecasting"
  Philippe Goulet Coulombe, arXiv: 2605.09712

Equations implemented:
  - Forecast Gain / Return: r_t = L^B_t - L^M_t  (Sec 2.1)
  - Forecast Sharpe Ratio  (Sec 2.2, Eq. Sharpe)
  - Forecast Sortino Ratio (Sec 2.2, Eq. Sortino)
  - Forecast Omega Ratio   (Sec 2.2, Eq. Omega)
  - Maximum Drawdown       (Sec 2.2, Eq. MaxDD)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def compute_returns(
    losses_benchmark: np.ndarray,
    losses_model: np.ndarray,
) -> np.ndarray:
    """
    Compute forecast gain (return) series.

    Paper: Section 2.1 — Forecast Gains and the Return Analogy
    Equation: r_t = L^B_t - L^M_t

    Args:
        losses_benchmark: Loss of the benchmark model at each period [T].
        losses_model:     Loss of the model under evaluation at each period [T].

    Returns:
        Return series {r_t} of shape [T]. Positive = model beats benchmark.
    """
    assert losses_benchmark.shape == losses_model.shape, (
        f"Shape mismatch: losses_benchmark {losses_benchmark.shape} "
        f"vs losses_model {losses_model.shape}"
    )
    return losses_benchmark - losses_model


class ForecastRiskMetrics:
    """
    Computes risk-adjusted performance metrics for a return series.

    Paper: Section 2.2 — Risk-Adjusted Forecast Metrics
    "Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

    Note: HAC adjustment is NOT applied to Sharpe/Sortino/Omega ratios.
    These are descriptive summaries of the realized gain distribution,
    not test statistics (see paper Section 2.2 for rationale).
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps: Small constant to avoid division by zero in denominators.
        """
        self.eps = eps

    def __repr__(self) -> str:
        return f"ForecastRiskMetrics(eps={self.eps})"

    def sharpe(self, returns: np.ndarray) -> float:
        """
        Forecast Sharpe Ratio.

        Paper: Sec 2.2, Eq: Sharpe = r_bar / s_r
        where s_r^2 = (1/(T-1)) * sum((r_t - r_bar)^2)

        Note: When returns are serially uncorrelated, Sharpe = DM / sqrt(T)
        (see Paper Section 2.3).

        Args:
            returns: Return series [T].

        Returns:
            Sharpe ratio scalar.
        """
        r_bar = np.mean(returns)
        s_r = np.std(returns, ddof=1)
        return float(r_bar / max(s_r, self.eps))

    def sortino(self, returns: np.ndarray) -> float:
        """
        Forecast Sortino Ratio.

        Paper: Sec 2.2, Eq: Sortino = r_bar / s_down
        where s_down = sqrt((1/T) * sum(min(r_t, 0)^2))

        The Sortino ratio penalizes only downside volatility, aligned with
        the asymmetric preferences of forecasters and policy makers.

        Args:
            returns: Return series [T].

        Returns:
            Sortino ratio scalar.
        """
        r_bar = np.mean(returns)
        # r_t^- = min(r_t, 0)
        r_minus = np.minimum(returns, 0.0)
        # s_down = sqrt(mean(r_minus^2))
        s_down = np.sqrt(np.mean(r_minus ** 2))
        return float(r_bar / max(s_down, self.eps))

    def omega(self, returns: np.ndarray) -> float:
        """
        Forecast Omega Ratio.

        Paper: Sec 2.2, Eq: Omega = sum(r+_t) / sum(|r-_t|)
        = Average Upside / Average Downside

        Non-parametric; does not impose distributional assumptions.
        Omega > 1: more upside than downside.

        Args:
            returns: Return series [T].

        Returns:
            Omega ratio scalar.
        """
        r_plus = np.maximum(returns, 0.0)
        r_minus = np.minimum(returns, 0.0)
        total_upside = np.sum(r_plus)
        total_downside = np.sum(np.abs(r_minus))
        return float(total_upside / max(total_downside, self.eps))

    def max_drawdown(self, returns: np.ndarray) -> float:
        """
        Maximum Drawdown.

        Paper: Sec 2.2, Eq:
          R_0 = 0, R_t = sum_{s=1}^t r_s
          M_t = max_{0<=u<=t} R_u
          DD_t = M_t - R_t
          MaxDD = max_{1<=t<=T} DD_t

        A model may display strong average performance while still
        experiencing prolonged episodes of underperformance.

        Args:
            returns: Return series [T].

        Returns:
            MaxDD as a positive scalar (magnitude of worst cumulative loss).
            Returns 0.0 if there is no drawdown.
        """
        # Cumulative gains: R_t = cumsum(r_s, s=1..t)
        R = np.concatenate([[0.0], np.cumsum(returns)])  # [T+1], R_0 = 0
        # Running maximum M_t = max_{0<=u<=t} R_u
        M = np.maximum.accumulate(R)
        # Drawdown at each point
        DD = M - R
        return float(np.max(DD))

    def all_metrics(
        self,
        losses_benchmark: np.ndarray,
        losses_model: np.ndarray,
        label: str = "model",
    ) -> dict:
        """
        Compute all risk-adjusted metrics from raw loss series.

        Args:
            losses_benchmark: Benchmark loss series [T].
            losses_model:     Model loss series [T].
            label:            Model label for the output dict.

        Returns:
            Dict with keys: label, T, return_mean, sharpe, sortino, omega, max_drawdown,
            autocorr_1 (first-order autocorrelation, reported as rho(1) in paper tables).
        """
        r = compute_returns(losses_benchmark, losses_model)
        T = len(r)

        # First-order autocorrelation (reported in paper tables as rho(1))
        if T > 2:
            rho1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        else:
            rho1 = float("nan")

        return {
            "label": label,
            "T": T,
            "return_mean": float(np.mean(r)),
            "sharpe": self.sharpe(r),
            "sortino": self.sortino(r),
            "omega": self.omega(r),
            "max_drawdown": -self.max_drawdown(r),  # Sign-flipped: negative = bad
            "autocorr_1": rho1,
        }
