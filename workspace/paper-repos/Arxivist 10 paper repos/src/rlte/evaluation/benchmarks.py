"""Heuristic benchmark execution strategies: Submit-and-Leave (SL) and
Time-Weighted Average Price (TWAP).

Paper reference: Section 3.4.
"""
from __future__ import annotations

import numpy as np


class SubmitAndLeave:
    """Submit the entire position as a single limit order at the best ask
    at t=0; hold thereafter. If inventory remains at T, it is force-sold by
    the environment's terminal market order (Section 3.3)."""

    def __init__(self, K: int):
        self.K = K

    def act(self, state: dict, step: int) -> np.ndarray:
        a = np.zeros(self.K + 1)
        if step == 0:
            a[1] = 1.0  # entire position at best ask (level 1)
        else:
            a[self.K] = 1.0  # hold (do nothing further)
        return a

    def __repr__(self) -> str:
        return "SubmitAndLeave()"


class TWAP:
    """Divide the position into N equal blocks of size M/N, posting a limit
    sell order of size M/N at the best ask at each decision step."""

    def __init__(self, K: int, N: int):
        self.K = K
        self.N = N

    def act(self, state: dict, step: int) -> np.ndarray:
        a = np.zeros(self.K + 1)
        remaining_steps = self.N - step
        # Allocate 1/remaining_steps of *current* inventory this step so the
        # schedule adapts if fills were partial (implementation detail not
        # specified exactly by the paper; ASSUMED even-split-of-remaining).
        frac = 1.0 / max(1, remaining_steps)
        a[1] = frac
        a[self.K] = 1.0 - frac
        return a

    def __repr__(self) -> str:
        return f"TWAP(N={self.N})"
