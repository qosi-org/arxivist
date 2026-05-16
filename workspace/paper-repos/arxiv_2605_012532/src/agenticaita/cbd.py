"""
cbd.py — Correlation-Break Diversification (Section 4.5)
Implements Equations 9–11 from arxiv:2605.12532 (Letteri 2026).

CBD operationalizes portfolio diversity within the Analyst's individual
reasoning. Assets decorrelated from BTC receive higher composite scores,
incentivizing idiosyncratic alpha selection.
"""
from __future__ import annotations
import math
import logging
from typing import Optional
import numpy as np

from .config import CBDConfig

logger = logging.getLogger(__name__)


class CorrelationBreakDiversification:
    """
    Correlation-Break Diversification (CBD) composite score.
    Section 4.5 of arxiv:2605.12532.

    Score Omega^a combines:
    - Exponentially-saturated anomaly magnitude z_tilde (Eq. 10)
    - BTC decorrelation metric rho_cb (Eq. 9)
    via convex combination (Eq. 11).
    """

    def __init__(self, cfg: CBDConfig) -> None:
        self.cfg = cfg
        # Price history buffer: asset -> deque of recent prices (for correlation)
        self._price_history: dict[str, list[float]] = {}
        self._btc_history: list[float] = []

    def update_price(self, asset: str, price: float, window: int) -> None:
        """Append price to rolling buffer for correlation computation."""
        if asset not in self._price_history:
            self._price_history[asset] = []
        self._price_history[asset].append(price)
        if len(self._price_history[asset]) > window:
            self._price_history[asset].pop(0)

    def update_btc_price(self, price: float, window: int) -> None:
        """Append BTC price to rolling buffer."""
        self._btc_history.append(price)
        if len(self._btc_history) > window:
            self._btc_history.pop(0)

    def compute_rho_cb(self, asset: str, window: int) -> float:
        """
        Eq. 9: rho_cb^a = 1 - |rho({p^a_tau}, {p^BTC_tau})|
        Decorrelation from BTC over rolling window W.
        Returns 0.5 (neutral) if insufficient history.

        NOTE: Correlation method (Pearson) ASSUMED — not specified in paper.
        """
        asset_prices = self._price_history.get(asset, [])
        btc_prices = self._btc_history

        min_len = min(len(asset_prices), len(btc_prices), window)
        if min_len < 5:  # Need at least 5 points for meaningful correlation
            logger.debug(f"Insufficient price history for {asset} CBD correlation, using 0.5")
            return 0.5  # Neutral: no decorrelation bonus, no penalty

        a = np.array(asset_prices[-min_len:])
        b = np.array(btc_prices[-min_len:])

        if self.cfg.correlation_method == "pearson":
            # ASSUMED: Pearson correlation (not specified in paper)
            corr_matrix = np.corrcoef(a, b)
            rho = float(corr_matrix[0, 1])
        elif self.cfg.correlation_method == "spearman":
            from scipy.stats import spearmanr
            rho, _ = spearmanr(a, b)
            rho = float(rho)
        else:
            raise ValueError(f"Unknown correlation method: {self.cfg.correlation_method}")

        if math.isnan(rho):
            return 0.5

        # Eq. 9: rho_cb = 1 - |rho|
        return 1.0 - abs(rho)

    def compute_z_tilde(self, z_score: float) -> float:
        """
        Eq. 10: z_tilde^a_t = (1 - exp(-kappa*(|z_t| - 2.0))) * 1[|z_t| >= 2.0]
        Exponential saturation mapping: maps unbounded Z-score into [0, 1).
        kappa=0.5 ensures z_tilde in [0,1) for all triggered observations.
        """
        abs_z = abs(z_score)
        if abs_z < self.cfg.z_score_threshold:  # threshold = 2.0
            return 0.0
        # Eq. 10
        return 1.0 - math.exp(-self.cfg.kappa * (abs_z - 2.0))

    def compute_omega(self, z_score: float, asset: str, window: int) -> float:
        """
        Eq. 11: Omega^a = alpha * z_tilde^a_t + (1 - alpha) * rho_cb^a
        Composite CBD score in [0, 1).
        alpha=0.5: equal weighting of anomaly informativeness and decorrelation.
        """
        z_tilde = self.compute_z_tilde(z_score)           # Eq. 10
        rho_cb = self.compute_rho_cb(asset, window)        # Eq. 9

        # Eq. 11: convex combination
        omega = self.cfg.alpha * z_tilde + (1.0 - self.cfg.alpha) * rho_cb

        logger.debug(f"CBD {asset}: z_tilde={z_tilde:.3f} rho_cb={rho_cb:.3f} Omega={omega:.3f}")
        return omega

    def __repr__(self) -> str:
        return f"CBD(alpha={self.cfg.alpha}, kappa={self.cfg.kappa}, method={self.cfg.correlation_method})"
