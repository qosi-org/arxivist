"""
evaluation/baseline.py — Heuristic Baseline Trading Strategy.

Implements the baseline described in Section 5:
  - When signal indicates UP: buy aggressively (at ask) until pos_max reached
  - When signal indicates DOWN: sell aggressively (at bid) until pos_min reached
  - When signal is NEUTRAL: place passive order to reduce position toward 0

"This heuristic utilises the same action space as the RL agent and yielded
better performance than trading using only passive or only aggressive orders."
(Section 5)

Paper: arXiv:2301.08688 — Section 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Action indices matching environment.py ACTION_MAP
SELL_AT_BID = 0   # aggressive sell (passive in book, marketable against bids)
SELL_AT_MID = 1
SELL_AT_ASK = 2   # crosses spread to sell
BUY_AT_BID  = 3   # crosses spread to buy
BUY_AT_MID  = 4
BUY_AT_ASK  = 5   # passive buy (resting at ask)
SKIP        = 6


class HeuristicBaseline:
    """Heuristic baseline trading strategy (Section 5).

    Trades aggressively in the direction of the signal until inventory limit
    is hit, then uses passive limit orders to slowly unwind when signal
    turns neutral.

    Args:
        pos_min: Minimum allowed inventory (default -10).
        pos_max: Maximum allowed inventory (default +10).
    """

    def __init__(self, pos_min: int = -10, pos_max: int = 10) -> None:
        self.pos_min = pos_min
        self.pos_max = pos_max

    def act(self, signal: NDArray, inventory: int) -> int:
        """Select action based on signal direction and current inventory.

        Args:
            signal: 3-vector [p_down, p_neutral, p_up] from SignalGenerator.
            inventory: Current stock inventory (integer).

        Returns:
            Action index (0–6).
        """
        direction = int(np.argmax(signal))  # 0=down, 1=neutral, 2=up

        if direction == 2:  # Signal says UP → buy aggressively
            if inventory < self.pos_max:
                return BUY_AT_ASK   # cross spread to build long position
            else:
                return SKIP
        elif direction == 0:  # Signal says DOWN → sell aggressively
            if inventory > self.pos_min:
                return SELL_AT_BID  # cross spread to build short position
            else:
                return SKIP
        else:  # Signal NEUTRAL → passively unwind position
            if inventory > 0:
                return SELL_AT_ASK  # passive sell to reduce long
            elif inventory < 0:
                return BUY_AT_BID   # passive buy to reduce short
            else:
                return SKIP

    def __repr__(self) -> str:
        return f"HeuristicBaseline(pos_bounds=[{self.pos_min},{self.pos_max}])"
