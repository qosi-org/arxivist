"""
forecast_risk.utils.dm_test
=============================
Diebold-Mariano test for equal predictive accuracy.

Implements the connection described in Section 2.3 of:
  "Quantifying the Risk-Return Tradeoff in Forecasting"
  Philippe Goulet Coulombe, arXiv: 2605.09712

The DM statistic uses a HAC estimator of the long-run variance:
  LRV_hat = gamma_0 + 2 * sum_{k=1}^K gamma_k

Key theoretical connection (Sec 2.3):
  When loss differentials are serially uncorrelated:
    DM = sqrt(T) * Sharpe

Reference: Diebold and Mariano (1995), Journal of Business & Economic Statistics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class DieboldMarianoTest:
    """
    Diebold-Mariano test for predictive accuracy.

    Paper: Section 2.3 — Connection to Diebold-Mariano
    "Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

    DM = r_bar / sqrt(LRV_hat / T)
    where LRV_hat = gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k (Newey-West truncation)

    Args:
        h: Forecast horizon (determines HAC lag truncation = h-1).
    """

    def __init__(self, h: int = 1):
        self.h = h

    def __repr__(self) -> str:
        return f"DieboldMarianoTest(h={self.h})"

    def _hac_lrv(self, d: np.ndarray) -> float:
        """
        HAC estimator of long-run variance with Bartlett kernel.

        LRV_hat = gamma_0 + 2 * sum_{k=1}^{K} (1 - k/(K+1)) * gamma_k
        where K = h - 1 (paper uses K = h-1 for direct h-step forecasts).

        Args:
            d: Loss differential series [T].

        Returns:
            Long-run variance estimate.
        """
        T = len(d)
        d_dm = d - np.mean(d)
        K = max(self.h - 1, 0)

        # gamma_0
        lrv = np.var(d, ddof=0)

        # Add autocovariances with Bartlett weights
        for k in range(1, K + 1):
            gamma_k = np.mean(d_dm[k:] * d_dm[:-k])
            bartlett_weight = 1.0 - k / (K + 1)
            lrv += 2 * bartlett_weight * gamma_k

        return max(lrv, 1e-12)  # guard against zero/negative variance

    def statistic(
        self,
        losses_a: np.ndarray,
        losses_b: np.ndarray,
    ) -> float:
        """
        Compute DM t-statistic.

        H0: E[L_a - L_b] = 0 (equal predictive accuracy)
        Positive DM → model B outperforms model A on average.

        Args:
            losses_a: Loss series for model A (benchmark) [T].
            losses_b: Loss series for model B (challenger) [T].

        Returns:
            DM t-statistic.
        """
        # d_t = L^A_t - L^B_t (positive when B beats A)
        d = losses_a - losses_b
        T = len(d)
        d_bar = np.mean(d)
        lrv = self._hac_lrv(d)
        se = np.sqrt(lrv / T)
        return float(d_bar / se)

    def pvalue(
        self,
        losses_a: np.ndarray,
        losses_b: np.ndarray,
        alternative: str = "two-sided",
    ) -> float:
        """
        Compute two-sided p-value for DM test using standard normal approximation.

        Args:
            losses_a:    Benchmark loss series [T].
            losses_b:    Challenger loss series [T].
            alternative: 'two-sided', 'greater', or 'less'.

        Returns:
            p-value.
        """
        dm = self.statistic(losses_a, losses_b)
        if alternative == "two-sided":
            return float(2 * (1 - stats.norm.cdf(abs(dm))))
        elif alternative == "greater":
            return float(1 - stats.norm.cdf(dm))
        elif alternative == "less":
            return float(stats.norm.cdf(dm))
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
