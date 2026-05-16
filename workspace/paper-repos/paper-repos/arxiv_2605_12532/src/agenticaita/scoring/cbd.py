"""
scoring/cbd.py — Correlation-Break Diversification (CBD) composite score.

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.5
Implements Eqs. 9-11 and Proposition 1.

Motivation: An agent evaluating assets independently tends to over-select correlated
positions, producing illusory diversification. CBD operationalizes portfolio diversity
within the Analyst's individual reasoning by rewarding assets that move independently
from BTC during statistically anomalous conditions.

Proposition 1 (Diversification incentive):
  For two triggered assets with identical z̃_t, the asset with higher ρ_cb
  receives a strictly higher Ω score. A perfect BTC-correlated asset (|ρ|=1)
  receives no diversification bonus.

Empirical result (5-day session):
  - Assets with ρ_cb > 0.85 (FARTCOIN, XPL, CC): mean PnL = +$5.06
  - Assets with ρ_cb < 0.15 (ETC, AVAX): mean PnL = -$2.86
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

from agenticaita.utils.config import CBDConfig

logger = logging.getLogger(__name__)

BTC_REFERENCE_ASSET = "BTC"


class CBD:
    """
    Correlation-Break Diversification composite score.

    Paper: Section 4.5, Eqs. 9-11.

    Computes Ω^a = α·z̃^a_t + (1-α)·ρ^a_cb where:
      ρ^a_cb = 1 - |corr(prices_asset, prices_BTC)|        (Eq. 9)
      z̃^a_t = (1 - exp(-κ(|z^a_t| - 2.0))) · 1[|z^a_t| ≥ 2.0]   (Eq. 10)
      Ω^a = α·z̃^a_t + (1-α)·ρ^a_cb, α=0.5                (Eq. 11)

    Both components are in [0,1), enabling equal-weight convex combination.

    Args:
        config: CBDConfig with alpha, kappa, window_bars.
    """

    def __init__(self, config: CBDConfig) -> None:
        self.config = config

    def __repr__(self) -> str:
        return f"CBD(alpha={self.config.alpha}, kappa={self.config.kappa}, W={self.config.window_bars})"

    def decorrelation_score(
        self,
        asset_prices: List[float],
        btc_prices: List[float],
    ) -> float:
        """
        Compute ρ^a_cb = 1 - |corr(prices_asset, prices_BTC)|.

        Eq. 9: decorrelation from BTC over the rolling window W.

        Args:
            asset_prices: Price series for the target asset, length = W.
            btc_prices: BTC price series over the same window, length = W.

        Returns:
            ρ_cb in [0, 1]: 0 = perfectly correlated with BTC, 1 = fully decorrelated.
        """
        if len(asset_prices) < 2 or len(btc_prices) < 2:
            # Insufficient data — assume maximum decorrelation (conservative)
            logger.debug("[CBD] insufficient price data for correlation; assuming rho_cb=1.0")
            return 1.0

        min_len = min(len(asset_prices), len(btc_prices))
        a = np.array(asset_prices[-min_len:])
        b = np.array(btc_prices[-min_len:])

        # Guard against zero-variance series
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            logger.debug("[CBD] near-zero variance in price series; assuming rho_cb=0.0")
            return 0.0

        rho = float(np.corrcoef(a, b)[0, 1])

        # Eq. 9: ρ^a_cb = 1 - |ρ|
        rho_cb = 1.0 - abs(rho)
        return float(np.clip(rho_cb, 0.0, 1.0))

    def normalized_anomaly(self, z_t: float) -> float:
        """
        Compute z̃^a_t: exponential saturation mapping of Z-score to [0,1).

        Eq. 10: z̃^a_t = (1 - exp(-κ(|z_t| - 2.0))) · 1[|z_t| ≥ 2.0]

        Maps unbounded |z_t| to [0,1) so it is commensurable with ρ_cb
        in the convex combination (Eq. 11). With κ=0.5, extreme outliers
        (|z_t| >> 2.0) do not structurally dominate the diversification signal.

        Args:
            z_t: Z-score of the asset's return magnitude.

        Returns:
            z̃_t in [0, 1).
        """
        if abs(z_t) < self.config.z_trigger_threshold if hasattr(self.config, 'z_trigger_threshold') else 2.0:
            return 0.0
        # Use the paper's threshold = 2.0
        excess = abs(z_t) - 2.0
        if excess <= 0.0:
            return 0.0
        # Eq. 10: (1 - exp(-kappa * excess))
        z_tilde = 1.0 - math.exp(-self.config.kappa * excess)
        return float(np.clip(z_tilde, 0.0, 1.0))

    def score(
        self,
        z_t: float,
        asset_prices: List[float],
        btc_prices: List[float],
    ) -> float:
        """
        Compute the full CBD composite score Ω^a.

        Eq. 11: Ω^a = α·z̃^a_t + (1-α)·ρ^a_cb

        The score is passed to the Analyst agent as context for deliberation.
        Per Case 2 in the paper, the Analyst explicitly cites Ω in its reasoning:
        'Active regime with positive composite score 0.83. High correlation break 0.85
        signals asset moving independently from BTC.'

        Args:
            z_t: Z-score of the asset's return (from AZTE).
            asset_prices: Rolling price buffer for target asset (length W).
            btc_prices: Rolling price buffer for BTC (length W).

        Returns:
            Ω^a in [0, 1): composite score for the Analyst.
        """
        rho_cb = self.decorrelation_score(asset_prices, btc_prices)
        z_tilde = self._normalized_anomaly_internal(z_t)

        # Eq. 11: convex combination
        omega = self.config.alpha * z_tilde + (1.0 - self.config.alpha) * rho_cb

        logger.debug(
            f"[CBD] z={z_t:.3f} → z̃={z_tilde:.3f}, rho_cb={rho_cb:.3f}, Ω={omega:.3f}"
        )
        return float(np.clip(omega, 0.0, 1.0))

    def _normalized_anomaly_internal(self, z_t: float) -> float:
        """Internal call — uses hardcoded 2.0 threshold per paper."""
        excess = abs(z_t) - 2.0
        if excess <= 0.0:
            return 0.0
        return float(np.clip(1.0 - math.exp(-self.config.kappa * excess), 0.0, 1.0))
