"""Background trading agents: NoiseTraders, TacticalTraders, StrategicTrader.

Paper reference: Section 5 (Market environment), Appendix A.1 (parameters).
"""
from __future__ import annotations

import numpy as np

from rlte.env.order_book import LimitOrderBook

# Intensities of limit order arrivals (lambda^{L,k}) and cancellations
# (lambda^{C,k}, already x10 in the paper's Table 3 -- values here are the
# true per-second intensities lambda^{C,k}) for noise traders, k=1..14
# (paper Table 3; higher levels ~0 and are omitted / treated as 0).
NOISE_LIMIT_INTENSITY = [
    0.2842, 0.5255, 0.2971, 0.2307, 0.0826, 0.0682, 0.0631, 0.0481,
    0.0462, 0.0321, 0.0178, 0.0015, 0.0001, 0.0000,
]
NOISE_CANCEL_INTENSITY = [v / 10.0 for v in [
    0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237,
    0.0233, 0.0178, 0.0127, 0.0012, 0.0001, 0.0000,
]]
NOISE_MARKET_INTENSITY = 0.1237  # lambda^M, Appendix A.1


def _poisson_count(rate: float, dt: float, rng: np.random.Generator) -> int:
    return int(rng.poisson(rate * dt))


def _shifted_half_normal_size(delta: float, rng: np.random.Generator, cap_std: float = 5.0) -> int:
    """Order size ~ 1 + delta*|Z|, Z~N(0,1), capped at cap_std standard
    deviations and rounded to nearest integer (Appendix A.1)."""
    z = abs(rng.standard_normal())
    z = min(z, cap_std)
    return max(1, round(1 + delta * z))


class NoiseTraders:
    """Independent-Poisson-process noise traders (Section 5.1)."""

    def __init__(self, D: int = 30, delta: float = 2.0):
        self.D = D
        self.delta = delta

    def step(self, book: LimitOrderBook, dt: float, rng: np.random.Generator,
              intensity_scale: float = 1.0) -> None:
        # Market orders (both directions).
        for side in ("buy", "sell"):
            n = _poisson_count(NOISE_MARKET_INTENSITY * intensity_scale, dt, rng)
            for _ in range(n):
                size = _shifted_half_normal_size(self.delta, rng)
                book.submit_market_order(side, size)

        # Limit orders and cancellations at each level, both sides.
        for k in range(1, min(self.D, len(NOISE_LIMIT_INTENSITY)) + 1):
            lim_rate = NOISE_LIMIT_INTENSITY[k - 1] * intensity_scale
            can_rate = NOISE_CANCEL_INTENSITY[k - 1] * intensity_scale
            for side in ("bid", "ask"):
                n_lim = _poisson_count(lim_rate, dt, rng)
                for _ in range(n_lim):
                    size = _shifted_half_normal_size(self.delta, rng)
                    book.submit_limit_order(side, k, size, owner="market")
                n_can = _poisson_count(can_rate, dt, rng)
                for _ in range(n_can):
                    size = _shifted_half_normal_size(self.delta, rng)
                    book.cancel_worst_queue_position(side, k, owner="market", size=size)


class TacticalTraders:
    """Order-book-imbalance-reactive traders (Section 5.2)."""

    def __init__(self, D: int = 30, delta: float = 2.0, c: float = 0.65,
                 dM: float = 2.0, dL: float = 2.0, dC: float = 2.0):
        self.D = D
        self.delta = delta
        self.c = c
        self.dM = dM
        self.dL = dL
        self.dC = dC

    def _imbalance(self, book: LimitOrderBook) -> float:
        bid_vols, ask_vols = book.get_volumes(self.D)
        weights = np.exp(-self.c * np.arange(self.D))
        V_b = float(np.dot(bid_vols, weights))
        V_a = float(np.dot(ask_vols, weights))
        denom = V_b + V_a
        if denom <= 0:
            return 0.0
        return (V_b - V_a) / denom

    def step(self, book: LimitOrderBook, dt: float, rng: np.random.Generator) -> None:
        I = self._imbalance(book)
        I_pos, I_neg = max(I, 0.0), max(-I, 0.0)

        # Market orders: buy intensity scales with I+, sell with I-.
        n_buy = _poisson_count(self.dM * I_pos, dt, rng)
        for _ in range(n_buy):
            book.submit_market_order("buy", _shifted_half_normal_size(self.delta, rng))
        n_sell = _poisson_count(self.dM * I_neg, dt, rng)
        for _ in range(n_sell):
            book.submit_market_order("sell", _shifted_half_normal_size(self.delta, rng))

        for k in range(1, self.D + 1):
            # Limit buy orders scale with I+, limit sell orders with I-.
            n_lb = _poisson_count(self.dL * I_pos, dt, rng)
            for _ in range(n_lb):
                book.submit_limit_order("bid", k, _shifted_half_normal_size(self.delta, rng), owner="market")
            n_ls = _poisson_count(self.dL * I_neg, dt, rng)
            for _ in range(n_ls):
                book.submit_limit_order("ask", k, _shifted_half_normal_size(self.delta, rng), owner="market")
            # Cancellations: buy cancels scale with I-, sell cancels with I+.
            n_cb = _poisson_count(self.dC * I_neg, dt, rng)
            for _ in range(n_cb):
                book.cancel_worst_queue_position("bid", k, owner="market",
                                                  size=_shifted_half_normal_size(self.delta, rng))
            n_cs = _poisson_count(self.dC * I_pos, dt, rng)
            for _ in range(n_cs):
                book.cancel_worst_queue_position("ask", k, owner="market",
                                                  size=_shifted_half_normal_size(self.delta, rng))


class StrategicTrader:
    """TWAP-style large trader buying or selling a position (Section 5.3)."""

    def __init__(self, nu_M: int = 1, nu_L: int = 2, dt_M: float = 3.0, dt_L: float = 3.0):
        self.nu_M = nu_M
        self.nu_L = nu_L
        self.dt_M = dt_M
        self.dt_L = dt_L
        self.direction: int | None = None  # +1 = buy, -1 = sell
        self._next_market_t = 0.0
        self._next_limit_t = 0.0

    def reset(self, rng: np.random.Generator, start_time: float) -> None:
        self.direction = 1 if rng.random() < 0.5 else -1
        self._next_market_t = start_time
        self._next_limit_t = start_time

    def step(self, book: LimitOrderBook, t: float, dt: float) -> None:
        assert self.direction is not None, "call reset() before step()"
        side = "buy" if self.direction == 1 else "sell"
        book_side = "ask" if side == "buy" else "bid"  # for a limit order joining the touch
        while self._next_market_t <= t:
            book.submit_market_order(side, self.nu_M)
            self._next_market_t += self.dt_M
        while self._next_limit_t <= t:
            book.submit_limit_order(book_side, 1, self.nu_L, owner="market")
            self._next_limit_t += self.dt_L
