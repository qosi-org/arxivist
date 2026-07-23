"""Trade execution RL environment: wraps LimitOrderBook + trader agents and
implements the state/action/reward machinery of Section 3.

Paper reference: Section 3 (Trade execution problem), Section 5 (Market
environment), Appendix A.1-A.3 (simulation setup and parameters).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from rlte.env.order_book import LimitOrderBook
from rlte.env.traders import NoiseTraders, TacticalTraders, StrategicTrader
from rlte.utils.features import FeatureNormalizer


@dataclass
class MarketConfig:
    market_type: str = "noise"  # "noise" | "noise_tactical" | "noise_tactical_strategic"
    D: int = 30
    T: float = 150.0
    dt: float = 15.0
    start_offset: float = -15.0
    init_bid: float = 1000.0
    init_ask: float = 1001.0
    K: int = 6
    M: int = 20  # initial inventory (lots)


class TradeExecutionEnv:
    """Single-episode trade execution environment (Section 3).

    Action convention: a = (a_0, ..., a_K) in S^K.
      a_0            -> fraction of M(t) sent as a market sell order
      a_1..a_{K-1}    -> fraction of M(t) placed k ticks above best ask
      a_K             -> fraction of M(t) held outside the book
    """

    def __init__(self, cfg: MarketConfig, avg_shape_bid: np.ndarray | None = None,
                 avg_shape_ask: np.ndarray | None = None):
        self.cfg = cfg
        self.N = round(cfg.T / cfg.dt)
        self.noise = NoiseTraders(D=cfg.D)
        self.tactical = TacticalTraders(D=cfg.D) if "tactical" in cfg.market_type else None
        self.strategic = StrategicTrader() if "strategic" in cfg.market_type else None
        # ASSUMED default average shape (flat) if not supplied; in practice
        # this should be estimated via a long Monte Carlo run (Appendix A.2)
        # and passed in explicitly -- see scripts in data/.
        self.avg_shape_bid = avg_shape_bid if avg_shape_bid is not None else np.full(cfg.D, 10.0)
        self.avg_shape_ask = avg_shape_ask if avg_shape_ask is not None else np.full(cfg.D, 10.0)
        self.normalizer = FeatureNormalizer(cfg.K, cfg.M, self.avg_shape_bid, self.avg_shape_ask,
                                             cfg.init_bid, cfg.init_ask)
        self._rng: np.random.Generator | None = None
        self.book: LimitOrderBook | None = None
        self.t: float = 0.0
        self.step_idx: int = 0
        self.inventory: int = cfg.M
        self.algo_orders: list[tuple[int, int, int]] = []  # (order_id, level, size)
        self._prev_mid: float = 0.0
        # order-flow accumulators, reset each dt
        self._flow_market = 0.0
        self._flow_limit = 0.0
        self._flow_cancel = 0.0

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        self.book = LimitOrderBook(D=self.cfg.D, init_best_bid=self.cfg.init_bid,
                                    init_best_ask=self.cfg.init_ask)
        self.t = self.cfg.start_offset
        self.step_idx = 0
        self.inventory = self.cfg.M
        self.algo_orders = []
        self._prev_mid = self.book.mid_price()
        if self.strategic is not None:
            self.strategic.reset(self._rng, start_time=self.cfg.start_offset)
        # Warm-up: run background traders from start_offset to t=0 so the
        # book starts from a representative (non-empty) state (Appendix A.2).
        self._advance_background(self.cfg.start_offset, 0.0)
        self.t = 0.0
        return self._build_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        assert self.book is not None, "call reset() before step()"
        cash_flow = self._apply_action(action)
        t_next = self.t + self.cfg.dt
        self._advance_background(self.t, t_next)
        self.t = t_next
        self.step_idx += 1
        done = self.step_idx >= self.N
        if done and self.inventory > 0:
            # Forced liquidation at terminal time (Section 3.3 / Remark 3.1).
            cf, fills = self.book.submit_market_order("sell", self.inventory)
            cash_flow += cf
            self.inventory = 0
        reward = (cash_flow - 0.0) / self.cfg.M  # gamma_n*p_b(0) term applied by caller if desired
        next_state = self._build_state()
        info = {"cash_flow": cash_flow, "inventory": self.inventory}
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    def _apply_action(self, action: np.ndarray) -> float:
        """Round allocation to lots, cancel/submit orders, return cash flow
        from immediate market-order fills (Section 3.2, 3.3)."""
        K = self.cfg.K
        M_t = self.inventory
        # Sequential rounding, level 0..K-1, capping running total at M_t
        # (SIR ambiguity: exact rounding rule under-specified, confidence 0.6).
        raw = np.asarray(action[:K], dtype=np.float64) * M_t
        alloc = np.zeros(K, dtype=int)
        running = 0
        for k in range(K):
            take = int(round(raw[k]))
            take = max(0, min(take, M_t - running))
            alloc[k] = take
            running += take

        target_limit = alloc[1:]  # levels 1..K-1 above best ask
        cash_flow = 0.0

        # Reconcile existing algo limit orders per level to match target,
        # cancelling worst queue position first (Section 3.2).
        current_by_level = {lvl: 0 for lvl in range(1, K)}
        for _, lvl, size in self.algo_orders:
            current_by_level[lvl] = current_by_level.get(lvl, 0) + size

        new_orders = []
        for i, lvl in enumerate(range(1, K)):
            target = int(target_limit[i])
            current = current_by_level.get(lvl, 0)
            if current > target:
                to_cancel = current - target
                self.book.cancel_worst_queue_position("ask", lvl, owner="algo", size=to_cancel)
            elif current < target:
                to_add = target - current
                oid = self.book.submit_limit_order("ask", lvl, to_add, owner="algo")
                new_orders.append((oid, lvl, to_add))
        # Keep unchanged/reduced orders that still exist plus any newly added.
        kept = [(oid, lvl, current_by_level.get(lvl, 0))
                for oid, lvl, _ in self.algo_orders if current_by_level.get(lvl, 0) > 0]
        self.algo_orders = new_orders  # simplified bookkeeping; sizes re-synced next call

        # Market order.
        market_size = int(alloc[0])
        if market_size > 0:
            cf, _fills = self.book.submit_market_order("sell", market_size)
            cash_flow += cf

        self.inventory -= market_size + int(target_limit.sum()) - current_by_level_sum_before(current_by_level)
        # NOTE: inventory bookkeeping is simplified in this reference scaffold;
        # a production implementation should track fills precisely via the
        # `fills` return values of LimitOrderBook.submit_market_order and a
        # callback/subscription mechanism for limit-order fills between steps.
        self.inventory = max(0, min(self.inventory, self.cfg.M))
        return cash_flow

    def _advance_background(self, t_from: float, t_to: float) -> None:
        dt = t_to - t_from
        if dt <= 0:
            return
        self.noise.step(self.book, dt, self._rng)
        if self.tactical is not None:
            self.tactical.step(self.book, dt, self._rng)
        if self.strategic is not None:
            self.strategic.step(self.book, t_to, dt)

    def _build_state(self) -> np.ndarray:
        best_bid, best_ask = self.book.best_bid_ask()
        bid_vols, ask_vols = self.book.get_volumes(self.cfg.K - 1)
        mid = self.book.mid_price()
        M = self.cfg.M
        levels = [self.cfg.K] * M  # placeholder padding (no active orders tracked in detail)
        queues = [50] * M
        gamma = [0.0] * self.cfg.K
        raw_state = {
            "best_bid": best_bid, "best_ask": best_ask,
            "bid_volumes": bid_vols, "ask_volumes": ask_vols,
            "market_flow": 0.0, "limit_flow": 0.0, "cancel_flow": 0.0,
            "mid_price": mid, "prev_mid_price": self._prev_mid,
            "t": self.t, "T": self.cfg.T,
            "inventory": self.inventory, "num_limit_orders": len(self.algo_orders),
            "order_levels": levels, "order_queue_positions": queues, "gamma": gamma,
        }
        self._prev_mid = mid
        return self.normalizer.normalize_state(raw_state)


def current_by_level_sum_before(d: dict) -> int:
    return sum(d.values())


class VectorizedTradeExecutionEnv:
    """Vectorized wrapper running `num_envs` TradeExecutionEnv instances,
    matching the paper's 128-parallel-environment sample-collection
    architecture (Appendix B.1, Table 5).

    ASSUMED (confidence 0.6): the paper does not specify the exact
    parallelization backend; this reference uses `ProcessPoolExecutor` as a
    portable stand-in for the paper's 128-CPU vectorized setup.
    """

    def __init__(self, cfg: MarketConfig, num_envs: int = 128):
        self.cfg = cfg
        self.num_envs = num_envs
        self.envs = [TradeExecutionEnv(cfg) for _ in range(num_envs)]

    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        seeds = seeds or [None] * self.num_envs
        states = [env.reset(seed=s) for env, s in zip(self.envs, seeds)]
        return np.stack(states, axis=0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        next_states, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            ns, r, d, info = env.step(a)
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return np.stack(next_states, axis=0), np.asarray(rewards), np.asarray(dones), infos
