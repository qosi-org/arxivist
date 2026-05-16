"""
azte.py — Adaptive Z-Score Trigger Engine (Section 4.1)
Implements Equations 1–3 from arxiv:2605.12532 (Letteri 2026).

The AZTE is the system's cognitive resource allocator. It distinguishes
market states rich in decision-relevant information from low-entropy noise
periods, activating the agent pipeline exclusively on statistically
anomalous market conditions.
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

from .schemas import TriggerEvent
from .config import AZTEConfig

logger = logging.getLogger(__name__)


@dataclass
class AssetBaseline:
    """Rolling baseline state for one asset."""
    prices: deque = field(default_factory=lambda: deque(maxlen=31))  # need prev price too
    returns: deque = field(default_factory=lambda: deque())
    last_trigger_ts: Optional[datetime] = None

    def add_return(self, r_t: float, window: int) -> None:
        self.returns.append(r_t)
        if len(self.returns) > window:
            self.returns.popleft()


class AdaptiveZScoreTriggerEngine:
    """
    Adaptive Z-Score Trigger Engine (AZTE).
    Section 4.1 of arxiv:2605.12532.

    Computes per-asset rolling Z-scores and fires TriggerEvents when
    Eq. 3 is satisfied:
        T_t = 1[z_t >= threshold] OR 1[r_t >= absolute_floor]

    Supports hot-restart via vol_history persistence (see EpisodicMemory).
    """

    def __init__(self, cfg: AZTEConfig, db=None) -> None:
        self.cfg = cfg
        self.db = db  # EpisodicMemory instance for persistence
        self._baselines: dict[str, AssetBaseline] = {}

    def _get_baseline(self, asset: str) -> AssetBaseline:
        if asset not in self._baselines:
            self._baselines[asset] = AssetBaseline()
        return self._baselines[asset]

    def compute_return_magnitude(self, p_t: float, p_prev: float) -> float:
        """
        Eq. 1: r_t = |p_t - p_{t-1}| / p_{t-1}
        Instantaneous return magnitude.
        """
        if p_prev == 0:
            return 0.0
        return abs((p_t - p_prev) / p_prev)

    def compute_z_score(self, r_t: float, returns: deque) -> Optional[float]:
        """
        Eq. 2: z_t = (r_t - mu_hat_W) / s_hat_W
        Returns None if not enough history or s_hat_W == 0.
        """
        if len(returns) < 2:
            return None
        arr = np.array(list(returns))
        mu = arr.mean()
        s = arr.std(ddof=1)
        if s < 1e-10:
            return None  # Numerically unstable; rely on absolute floor (Eq. 3)
        return float((r_t - mu) / s)

    def check_trigger(self, z_t: Optional[float], r_t: float) -> tuple[bool, str]:
        """
        Eq. 3: T_t = 1[z_t >= 2.0] OR 1[r_t >= 0.003]
        Returns (triggered: bool, triggered_by: str).
        """
        z_triggered = z_t is not None and z_t >= self.cfg.z_score_threshold
        r_triggered = r_t >= self.cfg.absolute_return_floor

        if z_triggered and r_triggered:
            return True, "both"
        elif z_triggered:
            return True, "z_score"
        elif r_triggered:
            return True, "absolute_floor"
        return False, ""

    def update(
        self,
        asset: str,
        price: float,
        timestamp: datetime,
        per_asset_cooldown_s: int = 300,
    ) -> Optional[TriggerEvent]:
        """
        Process one price observation for an asset.
        Returns a TriggerEvent if Eq. 3 fires, else None.
        Respects per-asset cooldown to prevent burst triggers.
        """
        baseline = self._get_baseline(asset)

        # Need at least one previous price
        if len(baseline.prices) == 0:
            baseline.prices.append(price)
            return None

        p_prev = baseline.prices[-1]
        baseline.prices.append(price)

        # Eq. 1: instantaneous return magnitude
        r_t = self.compute_return_magnitude(price, p_prev)
        baseline.add_return(r_t, self.cfg.rolling_window)

        # Eq. 2: Z-score
        z_t = self.compute_z_score(r_t, baseline.returns)

        # Eq. 3: disjunctive trigger
        triggered, triggered_by = self.check_trigger(z_t, r_t)

        if not triggered:
            return None

        # Per-asset cooldown check
        if baseline.last_trigger_ts is not None:
            elapsed = (timestamp - baseline.last_trigger_ts).total_seconds()
            if elapsed < per_asset_cooldown_s:
                logger.debug(f"{asset}: trigger suppressed by per-asset cooldown ({elapsed:.0f}s < {per_asset_cooldown_s}s)")
                return None

        baseline.last_trigger_ts = timestamp
        z_score_val = z_t if z_t is not None else 0.0

        logger.info(f"TRIGGER: {asset} z={z_score_val:.3f} r={r_t:.4f} via {triggered_by}")
        return TriggerEvent(
            asset=asset,
            timestamp=timestamp,
            z_score=z_score_val,
            return_magnitude=r_t,
            triggered_by=triggered_by,
        )

    async def load_history(self, asset: str, db) -> None:
        """
        Hot-restart: restore rolling baseline from vol_history.
        Eliminates warmup delay of W*60=30 minutes after container restart.
        Section 4.1.
        """
        rows = await db.get_vol_history(asset, limit=self.cfg.rolling_window)
        if not rows:
            return
        baseline = self._get_baseline(asset)
        for row in rows:
            baseline.add_return(row["r_t"], self.cfg.rolling_window)
            if row["price"]:
                baseline.prices.append(row["price"])
        logger.info(f"Hot-restart: loaded {len(rows)} history samples for {asset}")

    def __repr__(self) -> str:
        return f"AZTE(window={self.cfg.rolling_window}, threshold={self.cfg.z_score_threshold}σ, floor={self.cfg.absolute_return_floor})"
