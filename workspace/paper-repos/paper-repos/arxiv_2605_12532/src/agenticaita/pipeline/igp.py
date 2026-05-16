"""
pipeline/igp.py — Inference Gating Protocol (IGP).

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.4
Implements Definition 2: binary semaphore L serializing concurrent pipeline activations.

The IGP serves two roles:
  1. Prevents race conditions and interleaved database writes.
  2. Acts as a low-frequency filter: enforces minimum cognitive dwell time per decision,
     transforming micro-oscillations into activation opportunities for regime changes.

Paper log example:
  11:48:12 — Asset A: z=2.61 → PIPELINE START (lock acquired)
  11:48:13 — Asset B: z=2.30 → pipeline_busy, discarded
  11:49:10 — Asset A: complete (lock released)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from agenticaita.utils.config import IGPConfig

logger = logging.getLogger(__name__)


class IGP:
    """
    Inference Gating Protocol — mutex-based cognitive resource scheduler.

    Paper: Section 4.4, Definition 2.

    L ∈ {0,1}: binary semaphore initialized to 0.
    - Admit invocation iff L == 0
    - On admission: L ← 1
    - On completion: L ← 0
    - Concurrent trigger while L == 1: discard with logged pipeline_busy event.

    Global inter-pipeline cooldown (1800s) reinforces regularization beyond single-lock serialization.

    Args:
        config: IGPConfig with global_cooldown_s.
    """

    def __init__(self, config: IGPConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()        # L: binary semaphore equivalent
        self._last_release_time: Optional[datetime] = None
        self._busy_discards: int = 0
        self._total_admissions: int = 0

    def __repr__(self) -> str:
        return (
            f"IGP(cooldown={self.config.global_cooldown_s}s, "
            f"busy_discards={self._busy_discards}, "
            f"admissions={self._total_admissions})"
        )

    async def try_acquire(self, asset: str = "") -> bool:
        """
        Attempt to acquire the pipeline lock (Definition 2).

        Returns True if lock acquired (L was 0 → now 1).
        Returns False if lock is held (L == 1 → discard logged).

        Also enforces global_cooldown_s between pipeline completions.

        Args:
            asset: Asset requesting the lock (for logging only).
        """
        # Non-blocking lock check
        acquired = self._lock.locked() is False and self._lock.acquire  # quick check
        if self._lock.locked():
            self._busy_discards += 1
            logger.info(f"[IGP] pipeline_busy: {asset} discarded (lock held)")
            return False

        # Global cooldown check
        if self._last_release_time is not None:
            elapsed = (datetime.utcnow() - self._last_release_time).total_seconds()
            if elapsed < self.config.global_cooldown_s:
                remaining = self.config.global_cooldown_s - elapsed
                logger.info(
                    f"[IGP] global_cooldown: {asset} discarded "
                    f"({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)"
                )
                self._busy_discards += 1
                return False

        # Attempt non-blocking acquire
        acquired = self._lock.locked() is False
        if acquired:
            await self._lock.acquire()
            self._total_admissions += 1
            logger.info(f"[IGP] lock acquired for {asset} (admission #{self._total_admissions})")
        return acquired

    def release(self, asset: str = "") -> None:
        """
        Release the pipeline lock (L ← 0).

        Must be called after pipeline completion, even on error.
        """
        if self._lock.locked():
            self._lock.release()
            self._last_release_time = datetime.utcnow()
            logger.info(f"[IGP] lock released for {asset}")
        else:
            logger.warning(f"[IGP] release called but lock was not held (asset={asset})")

    @property
    def stats(self) -> dict:
        """Return IGP operational statistics."""
        return {
            "total_admissions": self._total_admissions,
            "busy_discards": self._busy_discards,
            "lock_held": self._lock.locked(),
            "last_release": self._last_release_time.isoformat() if self._last_release_time else None,
        }
