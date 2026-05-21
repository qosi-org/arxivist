"""
igp.py — Inference Gating Protocol (Section 4.4)
Implements the mutex-based cognitive resource scheduler from arxiv:2605.12532.

The IGP serializes concurrent agent activations, ensuring fully reproducible
audit trails and preventing race conditions on the LLM inference endpoint
and SQLite database.

Definition 2 (IGP Lock): binary semaphore L initialized to 0.
- Admitted iff L=0; upon admission L←1; upon completion L←0.
- Triggers arriving while L=1 are discarded with a logged pipeline_busy event.
"""
from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class InferenceGatingProtocol:
    """
    Mutex-based cognitive resource scheduler.
    Section 4.4 of arxiv:2605.12532.

    Implements Definition 2: binary semaphore protecting the SDP pipeline.
    Also enforces global cooldown between pipeline invocations.
    """

    def __init__(self, global_cooldown_s: int = 1800) -> None:
        self._lock = asyncio.Lock()
        self.global_cooldown_s = global_cooldown_s
        self._last_release_ts: Optional[float] = None
        self._lock_acquired_at: Optional[float] = None
        self.stats = {"admitted": 0, "busy_discarded": 0, "cooldown_discarded": 0}

    def is_locked(self) -> bool:
        """Return True if a pipeline is currently executing."""
        return self._lock.locked()

    def _cooldown_satisfied(self) -> bool:
        """Check global inter-pipeline cooldown (1800s / Table 2)."""
        if self._last_release_ts is None:
            return True
        elapsed = time.monotonic() - self._last_release_ts
        return elapsed >= self.global_cooldown_s

    async def acquire(self, asset: str = "") -> bool:
        """
        Attempt to acquire the IGP lock.
        Returns True if acquired (pipeline may proceed), False if busy or in cooldown.
        Non-blocking: does not wait.
        """
        # Check global cooldown first
        if not self._cooldown_satisfied():
            remaining = self.global_cooldown_s - (time.monotonic() - self._last_release_ts)
            logger.info(f"IGP: global cooldown active ({remaining:.0f}s remaining) — discarding {asset}")
            self.stats["cooldown_discarded"] += 1
            return False

        # Try to acquire without blocking (Definition 2: discard if L=1)
        acquired = self._lock.locked() is False and await self._try_acquire()
        if not acquired:
            logger.info(f"IGP: pipeline_busy — discarding trigger for {asset}")
            self.stats["busy_discarded"] += 1
            return False

        self._lock_acquired_at = time.monotonic()
        self.stats["admitted"] += 1
        logger.info(f"IGP: lock acquired for {asset}")
        return True

    async def _try_acquire(self) -> bool:
        """Non-blocking lock acquisition attempt."""
        try:
            return self._lock.acquire_nowait()  # type: ignore[attr-defined]
        except AttributeError:
            # asyncio.Lock doesn't have acquire_nowait in all versions
            # Fallback: check locked state then acquire with timeout=0
            if self._lock.locked():
                return False
            await asyncio.wait_for(self._lock.acquire(), timeout=0.001)
            return True
        except (asyncio.TimeoutError, Exception):
            return False

    async def release(self) -> None:
        """Release the IGP lock after pipeline completion."""
        if self._lock.locked():
            self._lock.release()
            self._last_release_ts = time.monotonic()
            duration = time.monotonic() - (self._lock_acquired_at or time.monotonic())
            logger.info(f"IGP: lock released (pipeline duration: {duration:.1f}s)")

    def get_stats(self) -> dict:
        return dict(self.stats)

    def __repr__(self) -> str:
        return f"IGP(locked={self.is_locked()}, cooldown={self.global_cooldown_s}s, stats={self.stats})"
