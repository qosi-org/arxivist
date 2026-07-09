"""Discrete-price limit order book with FIFO queues per price level.

Paper reference: Section 2 (Limit order books), Figure 1 worked example,
Appendix A.1 (D=30 visible levels per side).
"""
from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Order:
    order_id: int
    side: str  # "bid" or "ask"
    level: int  # 1 = best price, k = k-1 ticks worse than best
    size: int
    owner: str  # "algo" or "market" (background traders)


class LimitOrderBook:
    """Discrete limit order book with D visible price levels per side.

    Levels are 1-indexed: level 1 is the best bid/ask, level k is k-1 ticks
    away from the best price on that side (Section 2, Eq. (1)).
    """

    def __init__(self, tick_size: float = 1.0, D: int = 30,
                 init_best_bid: float = 1000.0, init_best_ask: float = 1001.0):
        self.tick_size = tick_size
        self.D = D
        self.best_bid_price = init_best_bid
        self.best_ask_price = init_best_ask
        # queues[side][level] -> deque[Order], FIFO (front = first in queue)
        self.queues: dict[str, list[deque[Order]]] = {
            "bid": [deque() for _ in range(D + 1)],  # index 0 unused
            "ask": [deque() for _ in range(D + 1)],
        }
        self._id_counter = itertools.count(1)
        self._order_index: dict[int, tuple[str, int]] = {}  # order_id -> (side, level)

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------
    def best_bid_ask(self) -> tuple[float, float]:
        return self.best_bid_price, self.best_ask_price

    def mid_price(self) -> float:
        return 0.5 * (self.best_bid_price + self.best_ask_price)

    def get_volumes(self, k: int) -> tuple[list[int], list[int]]:
        """Return (bid_volumes, ask_volumes) for levels 1..k (Eq. 1)."""
        bid_vols = [sum(o.size for o in self.queues["bid"][lvl]) for lvl in range(1, k + 1)]
        ask_vols = [sum(o.size for o in self.queues["ask"][lvl]) for lvl in range(1, k + 1)]
        return bid_vols, ask_vols

    # ------------------------------------------------------------------
    # Order actions
    # ------------------------------------------------------------------
    def submit_limit_order(self, side: str, level: int, size: int, owner: str = "market") -> int:
        """Insert a limit order at the back of the queue for (side, level)."""
        assert side in ("bid", "ask")
        assert 1 <= level <= self.D
        order_id = next(self._id_counter)
        order = Order(order_id, side, level, size, owner)
        self.queues[side][level].append(order)
        self._order_index[order_id] = (side, level)
        return order_id

    def cancel_order(self, order_id: int, size: int | None = None) -> bool:
        """Cancel (fully or partially) a resting order by id."""
        if order_id not in self._order_index:
            return False
        side, level = self._order_index[order_id]
        q = self.queues[side][level]
        for o in q:
            if o.order_id == order_id:
                if size is None or size >= o.size:
                    q.remove(o)
                    del self._order_index[order_id]
                else:
                    o.size -= size
                return True
        return False

    def cancel_worst_queue_position(self, side: str, level: int, owner: str, size: int) -> int:
        """Cancel up to `size` lots belonging to `owner`, worst queue position
        first (i.e. from the back of the FIFO queue), per Section 3.2's stated
        cancellation priority rule.

        Returns the number of lots actually cancelled.
        """
        q = self.queues[side][level]
        cancelled = 0
        for o in reversed(list(q)):
            if o.owner != owner:
                continue
            take = min(size - cancelled, o.size)
            if take <= 0:
                continue
            if take == o.size:
                q.remove(o)
                self._order_index.pop(o.order_id, None)
            else:
                o.size -= take
            cancelled += take
            if cancelled >= size:
                break
        return cancelled

    def submit_market_order(self, side: str, size: int) -> tuple[float, list[tuple[str, int]]]:
        """Match a market order of `size` lots against the opposite side,
        walking price levels from best to worst (Section 2).

        `side` is the side of the *incoming* market order ("buy" hits asks,
        "sell" hits bids).

        Returns:
            cash_flow: signed cash flow to the *submitter* of this market
                order (positive for a sell, negative for a buy).
            fills: list of (owner, filled_size) for bookkeeping which of the
                algorithm's resting limit orders were filled.
        """
        assert side in ("buy", "sell")
        book_side = "ask" if side == "buy" else "bid"
        remaining = size
        cash_flow = 0.0
        fills: list[tuple[str, int]] = []
        level = 1
        while remaining > 0 and level <= self.D:
            q = self.queues[book_side][level]
            price = (self.best_ask_price if book_side == "ask" else self.best_bid_price) \
                + (level - 1) * self.tick_size * (1 if book_side == "ask" else -1)
            while remaining > 0 and q:
                o = q[0]
                take = min(remaining, o.size)
                fills.append((o.owner, take))
                signed = take if side == "sell" else -take
                cash_flow += signed * price
                o.size -= take
                remaining -= take
                if o.size == 0:
                    q.popleft()
                    self._order_index.pop(o.order_id, None)
            level += 1
        self._refresh_best_prices()
        return cash_flow, fills

    def _refresh_best_prices(self) -> None:
        """Advance best_bid/best_ask if the front level(s) are now empty."""
        while self.queues["bid"][1].__len__() == 0 and self._level_has_any_deeper("bid"):
            self._shift_levels("bid")
        while self.queues["ask"][1].__len__() == 0 and self._level_has_any_deeper("ask"):
            self._shift_levels("ask")

    def _level_has_any_deeper(self, side: str) -> bool:
        return any(len(self.queues[side][lvl]) > 0 for lvl in range(2, self.D + 1))

    def _shift_levels(self, side: str) -> None:
        """Shift queues down by one level and update best price by one tick."""
        for lvl in range(1, self.D):
            self.queues[side][lvl] = self.queues[side][lvl + 1]
            for o in self.queues[side][lvl]:
                o.level = lvl
                self._order_index[o.order_id] = (side, lvl)
        self.queues[side][self.D] = deque()
        if side == "bid":
            self.best_bid_price -= self.tick_size
        else:
            self.best_ask_price += self.tick_size

    def __repr__(self) -> str:
        b, a = self.best_bid_ask()
        return f"LimitOrderBook(best_bid={b}, best_ask={a})"
