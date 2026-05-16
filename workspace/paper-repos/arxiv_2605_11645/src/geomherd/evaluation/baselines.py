"""
src/geomherd/evaluation/baselines.py
Classical herding baseline detectors: LSV (1992) and CSAD/CCK (2000).
Paper: arXiv:2605.11645, Section 3.2 and Section 3.3.2 (Eq. 8)

Implements:
  - LSVBaseline: Lakonishok-Shleifer-Vishny (1992) buy/sell imbalance
  - CSADBaseline: Chang-Cheng-Khorana (2000) cross-sectional absolute deviation
  - Augmented CCK regression (Eq. 8): adds kappa_bar_OR as a third regressor
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class LSVBaseline:
    """
    Lakonishok-Shleifer-Vishny (1992) institutional herding statistic.

    Paper reference: Section 3.2, Table 3
        LSV statistic measures buy/sell imbalance relative to chance.
        In simulator: computed from windowed buy/sell flows per asset/agent group.
        Achieves recall=1.00 on both supercritical and subcritical (fires on all),
        so not a regime classifier (FAR_sub=1.00).

    Reference: Lakonishok, Shleifer, Vishny (1992), J. Financial Economics 32(1):23-43.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._buy_buffer: List[float] = []
        self._sell_buffer: List[float] = []

    def update(self, actions: np.ndarray) -> float:
        """
        Update with current-step actions and return rolling LSV statistic.

        Args:
            actions: [N] int array with values {0=buy, 1=hold, 2=sell}
        Returns:
            lsv: LSV herding measure (float in [0, 1])
        """
        N = len(actions)
        buy_frac = float((actions == 0).sum()) / N
        sell_frac = float((actions == 2).sum()) / N
        self._buy_buffer.append(buy_frac)
        self._sell_buffer.append(sell_frac)
        if len(self._buy_buffer) > self.window:
            self._buy_buffer.pop(0)
            self._sell_buffer.pop(0)
        if len(self._buy_buffer) < 2:
            return 0.0
        p_bar = float(np.mean(self._buy_buffer))  # expected buy fraction
        # LSV: |buy_frac - p_bar| + |sell_frac - (1-p_bar)| - correction
        # Simplified: mean absolute deviation from expected fractions
        lsv_vals = [abs(b - p_bar) for b in self._buy_buffer]
        return float(np.mean(lsv_vals))

    @staticmethod
    def compute_static(buy_fracs: np.ndarray, sell_fracs: np.ndarray) -> np.ndarray:
        """
        Compute LSV statistic over a full trajectory array.

        Args:
            buy_fracs: [T] array of buy fractions per step
            sell_fracs: [T] array of sell fractions per step
        Returns:
            lsv: [T] array of LSV statistics
        """
        p_bar = buy_fracs.mean()
        lsv = np.abs(buy_fracs - p_bar) + np.abs(sell_fracs - (1 - p_bar))
        return lsv

    def reset(self) -> None:
        self._buy_buffer.clear()
        self._sell_buffer.clear()

    def __repr__(self) -> str:
        return f"LSVBaseline(window={self.window})"


class CSADBaseline:
    """
    Chang-Cheng-Khorana (2000) Cross-Sectional Absolute Deviation herding measure.

    Paper reference: Section 3.2, Section 3.3.2 (Eq. 8)
        CSAD_t = E_i[|r_{i,t} - r_bar_t|]
        Augmented CCK regression (Eq. 8):
            CSAD_t = alpha + gamma1*|Rm_t| + gamma2*Rm_t^2 + gamma3*kappa_bar_OR(t) + eps_t
        Estimated with HAC Newey-West standard errors.

    Reference: Chang, Cheng, Khorana (2000), J. Banking & Finance 24(10):1651-1679.
    """

    @staticmethod
    def compute(returns: np.ndarray) -> np.ndarray:
        """
        Compute CSAD over a trajectory of per-agent returns.

        Paper: Assumption A5 — CSAD_t = E_i[|r_{i,t} - r_bar_t|]

        Args:
            returns: [T, N] or [T] array of per-agent (or market) returns
        Returns:
            csad: [T] array of CSAD values
        """
        if returns.ndim == 1:
            # Single asset / market return: CSAD = 0 (degenerate case)
            return np.zeros(len(returns))
        # [T, N] case: cross-sectional absolute deviation at each t
        r_bar = returns.mean(axis=1, keepdims=True)  # [T, 1] market mean return
        csad = np.abs(returns - r_bar).mean(axis=1)  # [T]
        return csad

    @staticmethod
    def cck_regression(
        csad: np.ndarray,
        rm: np.ndarray,
        kappa_or: Optional[np.ndarray] = None,
        use_hac: bool = True,
        hac_lags: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Augmented CCK regression with optional GeomHerd term.

        Standard CCK:   CSAD_t = alpha + gamma1*|Rm_t| + gamma2*Rm_t^2 + eps
        Augmented (Eq. 8): adds gamma3 * kappa_bar_OR(t)

        Paper reference: Section 3.3.2, Eq. 8
            gamma3 median = -0.0072, CI = [-0.00769, -0.00602]

        Args:
            csad: [T] CSAD time series
            rm: [T] market return time series
            kappa_or: [T] optional kappa_bar_OR time series (adds gamma3 term if given)
            use_hac: Use HAC Newey-West standard errors (default True)
            hac_lags: Number of HAC lags (None = automatic, Newey-West rule)
        Returns:
            Dict with coefficients alpha, gamma1, gamma2, (gamma3 if kappa_or given),
            and their standard errors and p-values.
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError("statsmodels required for CCK regression. pip install statsmodels")

        T = len(csad)
        X = np.column_stack([
            np.ones(T),        # alpha (intercept)
            np.abs(rm),        # |Rm_t| (gamma1)
            rm ** 2,           # Rm_t^2 (gamma2)
        ])
        if kappa_or is not None:
            X = np.column_stack([X, kappa_or])  # gamma3

        model = sm.OLS(csad, X)
        if use_hac:
            # Newey-West HAC standard errors
            # Automatic bandwidth: lags = floor(4*(T/100)^(2/9)) (Newey-West rule)
            if hac_lags is None:
                hac_lags = int(4 * (T / 100) ** (2 / 9))
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
        else:
            results = model.fit()

        coef_names = ["alpha", "gamma1", "gamma2"]
        if kappa_or is not None:
            coef_names.append("gamma3")

        output = {}
        for i, name in enumerate(coef_names):
            output[f"{name}_coef"] = float(results.params[i])
            output[f"{name}_se"] = float(results.bse[i])
            output[f"{name}_pval"] = float(results.pvalues[i])
        output["r_squared"] = float(results.rsquared)
        output["n_obs"] = T
        return output

    @staticmethod
    def rolling_csad(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling CSAD with given window (for streaming detection)."""
        T = returns.shape[0] if returns.ndim > 1 else len(returns)
        csad_full = CSADBaseline.compute(returns)
        result = np.full(T, np.nan)
        for t in range(window - 1, T):
            result[t] = csad_full[t - window + 1:t + 1].mean()
        return result
