"""
Limit order book (LOB) engine.

Implements the market microstructure described in Section 2.1 of the paper:
a continuous double-auction limit order book supporting both market and
limit orders, with a tick size fixed at 1 basis point of the initial price.

IMPLEMENTATION ASSUMPTION (SIR confidence 0.6): the exact matching-engine
internals (cancellation policy beyond HFT replenishment, tie-breaking rule)
are not specified in the paper. We use a standard FIFO price-time-priority
book, which is common practice for minimal LOB ABMs of this kind. This
choice affects the emergent depth-profile exponent gamma reported in
Section 4.4 / 5.2 of the paper — gamma is *not* hard-coded, it is measured
from whatever book shape this engine produces, so any sensitivity to this
assumption is visible in the results rather than silently baked in.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Side = Literal["buy", "sell"]


@dataclass
class Trade:
    """A single executed trade.

    Args:
        t: Simulation step at which the trade executed.
        price: Execution price.
        size: Executed size (always positive).
        buy_agent_id: Agent id of the buy-side counterparty.
        sell_agent_id: Agent id of the sell-side counterparty.
        aggressor_side: Which side initiated the trade ("buy" or "sell").
        aggressor_agent_id: Agent id of the aggressor (used for metaorder
            reconstruction, Section 2.5).
    """

    t: int
    price: float
    size: float
    buy_agent_id: int
    sell_agent_id: int
    aggressor_side: Side
    aggressor_agent_id: int


@dataclass
class _RestingOrder:
    order_id: int
    side: Side
    price: float
    size: float
    agent_id: int
    t_submitted: int


class LimitOrderBook:
    """A minimal continuous double-auction limit order book.

    Args:
        initial_price: Starting mid-price for the book.
        tick_size_bp: Tick size expressed in basis points of initial_price
            (Section 2.1: "tick size is set to 1 basis point of the initial
            price").
    """

    def __init__(self, initial_price: float, tick_size_bp: float = 1.0):
        assert initial_price > 0, f"initial_price must be positive, got {initial_price}"
        self.initial_price = initial_price
        self.tick_size = initial_price * tick_size_bp / 10_000.0

        # price (rounded to tick grid) -> list[_RestingOrder], FIFO within a price level
        self._bids: dict[float, list[_RestingOrder]] = {}
        self._asks: dict[float, list[_RestingOrder]] = {}
        self._order_id_counter = itertools.count(1)
        self._orders_by_id: dict[int, _RestingOrder] = {}

        self.last_trade_price: float = initial_price
        self.trade_tape: list[Trade] = []

    # ------------------------------------------------------------------ #
    # Price grid helpers
    # ------------------------------------------------------------------ #
    def _round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def best_bid(self) -> float | None:
        return max(self._bids.keys()) if self._bids else None

    def best_ask(self) -> float | None:
        return min(self._asks.keys()) if self._asks else None

    def best_bid_ask(self) -> tuple[float | None, float | None]:
        """Returns (best_bid, best_ask)."""
        return self.best_bid(), self.best_ask()

    def mid_price(self) -> float:
        """Returns the mid price, falling back to last trade price if one side is empty."""
        bb, ba = self.best_bid_ask()
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return self.last_trade_price

    def depth_profile(self, side: Side, n_levels: int = 50) -> np.ndarray:
        """Cumulative depth V(q) away from the best price on `side`.

        Used to estimate the LOB-walking exponent gamma via V(q) ~ q^(1+gamma)
        (Section 4.4, Eq. 5 context).

        Args:
            side: "buy" (bid side) or "sell" (ask side).
            n_levels: Number of price levels to include.

        Returns:
            1D array of length n_levels: cumulative size from best price
            outward.
        """
        book = self._bids if side == "buy" else self._asks
        best = self.best_bid() if side == "buy" else self.best_ask()
        if best is None or not book:
            return np.zeros(n_levels)

        levels = sorted(book.keys(), reverse=(side == "buy"))[:n_levels]
        sizes = np.array([sum(o.size for o in book[p]) for p in levels])
        return np.cumsum(sizes)

    # ------------------------------------------------------------------ #
    # Order submission
    # ------------------------------------------------------------------ #
    def submit_limit_order(
        self, side: Side, price: float, size: float, agent_id: int, t: int
    ) -> int:
        """Submit a limit (resting) order.

        Args:
            side: "buy" or "sell".
            price: Limit price (rounded to the tick grid).
            size: Order size (must be positive).
            agent_id: Submitting agent's id.
            t: Current simulation step.

        Returns:
            The new order's integer id.
        """
        assert size > 0, f"Limit order size must be positive, got {size}"
        price = self._round_to_tick(price)
        order_id = next(self._order_id_counter)
        order = _RestingOrder(order_id, side, price, size, agent_id, t)
        book = self._bids if side == "buy" else self._asks
        book.setdefault(price, []).append(order)
        self._orders_by_id[order_id] = order
        return order_id

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order by id. Returns True if it was found and removed."""
        order = self._orders_by_id.pop(order_id, None)
        if order is None:
            return False
        book = self._bids if order.side == "buy" else self._asks
        level = book.get(order.price)
        if level is None:
            return False
        try:
            level.remove(order)
        except ValueError:
            return False
        if not level:
            del book[order.price]
        return True

    def submit_market_order(
        self, side: Side, size: float, agent_id: int, t: int
    ) -> list[Trade]:
        """Submit a market order that walks the opposite side of the book.

        Args:
            side: "buy" (consumes asks) or "sell" (consumes bids).
            size: Order size to execute (must be positive).
            agent_id: Aggressor agent's id.
            t: Current simulation step.

        Returns:
            List of Trade objects generated by this order (may be empty if
            the book is empty on the opposite side).
        """
        assert size > 0, f"Market order size must be positive, got {size}"
        book = self._asks if side == "buy" else self._bids
        remaining = size
        trades: list[Trade] = []

        while remaining > 1e-12 and book:
            best_price = min(book.keys()) if side == "buy" else max(book.keys())
            level = book[best_price]
            while remaining > 1e-12 and level:
                resting = level[0]
                fill_size = min(remaining, resting.size)

                if side == "buy":
                    buy_agent, sell_agent = agent_id, resting.agent_id
                else:
                    buy_agent, sell_agent = resting.agent_id, agent_id

                trade = Trade(
                    t=t,
                    price=best_price,
                    size=fill_size,
                    buy_agent_id=buy_agent,
                    sell_agent_id=sell_agent,
                    aggressor_side=side,
                    aggressor_agent_id=agent_id,
                )
                trades.append(trade)
                self.trade_tape.append(trade)
                self.last_trade_price = best_price

                resting.size -= fill_size
                remaining -= fill_size

                if resting.size <= 1e-12:
                    level.pop(0)
                    self._orders_by_id.pop(resting.order_id, None)
            if not level:
                del book[best_price]

        return trades

    def n_resting_orders(self) -> int:
        """Total number of resting orders currently in the book (both sides)."""
        return len(self._orders_by_id)

    def __repr__(self) -> str:
        bb, ba = self.best_bid_ask()
        return f"LimitOrderBook(mid={self.mid_price():.2f}, best_bid={bb}, best_ask={ba})"
