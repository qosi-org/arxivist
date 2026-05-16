"""
trigger/azte.py — Adaptive Z-Score Trigger Engine (AZTE).

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.1
Implements Eqs. 1–3: rolling return magnitude, Z-score baseline, disjunctive trigger.

The AZTE is the system's cognitive resource allocator — it gates LLM inference
exclusively on statistically anomalous market conditions.

Key property: REGIME INVARIANCE — the 2σ threshold self-adapts to prevailing
volatility without manual recalibration across market regimes.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from agenticaita.pipeline.contracts import TriggerEvent
from agenticaita.utils.config import TriggerConfig

logger = logging.getLogger(__name__)


class AZTE:
    """
    Adaptive Z-Score Trigger Engine.

    Paper: Section 4.1, Eqs. 1-3.

    Computes per-asset rolling return magnitudes and fires trigger events when:
      - z_t >= z_threshold  (statistical anomaly gate)       Eq. 3
      - r_t >= r_floor      (absolute return safety net)     Eq. 3

    Supports hot-restart: rolling buffers can be populated from vol_history DB,
    eliminating up to W * polling_interval = 30min warmup delay.

    Args:
        config: TriggerConfig with all AZTE hyperparameters.
    """

    def __init__(self, config: TriggerConfig) -> None:
        self.config = config
        self._price_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.window_bars)
        )
        self._return_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.window_bars)
        )
        self._last_prices: Dict[str, float] = {}
        self._last_trigger_time: Dict[str, Optional[datetime]] = defaultdict(lambda: None)

    def __repr__(self) -> str:
        return (
            f"AZTE(z_thresh={self.config.z_threshold}, r_floor={self.config.r_floor}, "
            f"W={self.config.window_bars}, cooldown={self.config.per_asset_cooldown_s}s)"
        )

    def hot_restart(self, asset: str, historical_returns: list[float], historical_prices: list[float]) -> None:
        """
        Restore rolling buffers from persisted vol_history.

        Paper: Section 4.1 — "enabling hot restart: after container failure, the
        baseline is restored immediately, eliminating a warmup delay of up to
        W × 60 = 30 minutes."

        Args:
            asset: Asset symbol.
            historical_returns: List of past return magnitudes (r_t values), most recent last.
            historical_prices: List of past close prices, most recent last.
        """
        W = self.config.window_bars
        for r in historical_returns[-W:]:
            self._return_buffers[asset].append(r)
        for p in historical_prices[-W:]:
            self._price_buffers[asset].append(p)
        if historical_prices:
            self._last_prices[asset] = historical_prices[-1]
        logger.info(f"[AZTE] hot_restart {asset}: loaded {len(self._return_buffers[asset])} return samples")

    async def update(self, asset: str, price: float) -> Optional[TriggerEvent]:
        """
        Process a new price tick for an asset.

        Implements Eqs. 1-3:
          r_t = |p_t - p_{t-1}| / p_{t-1}          (Eq. 1)
          z_t = (r_t - mu_hat_W) / s_hat_W          (Eq. 2)
          T_t = 1[z_t >= 2.0] OR 1[r_t >= 0.003]   (Eq. 3)

        Args:
            asset: Asset symbol.
            price: Current price p_t.

        Returns:
            TriggerEvent if T_t = 1 and cooldown has elapsed, else None.
        """
        if asset not in self._last_prices:
            self._last_prices[asset] = price
            self._price_buffers[asset].append(price)
            return None

        p_prev = self._last_prices[asset]
        self._last_prices[asset] = price
        self._price_buffers[asset].append(price)

        # Eq. 1: instantaneous return magnitude
        if p_prev == 0.0:
            return None
        r_t = abs(price - p_prev) / p_prev

        # Update rolling return buffer
        self._return_buffers[asset].append(r_t)

        # Need at least 2 samples for a meaningful Z-score
        if len(self._return_buffers[asset]) < 2:
            return None

        returns = np.array(self._return_buffers[asset])
        mu_hat = float(np.mean(returns))
        s_hat = float(np.std(returns, ddof=1))

        # Eq. 2: rolling Z-score (handle near-zero std)
        if s_hat > 1e-10:
            z_t = (r_t - mu_hat) / s_hat
        else:
            # Standard deviation → 0: fall back to absolute return gate only
            z_t = 0.0

        # Eq. 3: disjunctive trigger
        triggered_by_z = z_t >= self.config.z_threshold
        triggered_by_r = r_t >= self.config.r_floor

        if not (triggered_by_z or triggered_by_r):
            return None

        # Per-asset cooldown check
        now = datetime.utcnow()
        last = self._last_trigger_time[asset]
        if last is not None:
            elapsed = (now - last).total_seconds()
            if elapsed < self.config.per_asset_cooldown_s:
                logger.debug(
                    f"[AZTE] {asset}: trigger suppressed by cooldown "
                    f"({elapsed:.0f}s < {self.config.per_asset_cooldown_s}s)"
                )
                return None

        self._last_trigger_time[asset] = now

        if triggered_by_z and triggered_by_r:
            by = "both"
        elif triggered_by_z:
            by = "z_score"
        else:
            by = "r_floor"

        event = TriggerEvent(
            asset=asset,
            timestamp=now,
            z_score=z_t,
            return_magnitude=r_t,
            triggered_by=by,
        )
        logger.info(f"[AZTE] TRIGGER {asset}: z={z_t:.3f}, r={r_t:.4f}, by={by}")
        return event

    def get_return_buffer(self, asset: str) -> list[float]:
        """Return a copy of the rolling return buffer for an asset."""
        return list(self._return_buffers[asset])

    def get_price_buffer(self, asset: str) -> list[float]:
        """Return a copy of the rolling price buffer for an asset."""
        return list(self._price_buffers[asset])
