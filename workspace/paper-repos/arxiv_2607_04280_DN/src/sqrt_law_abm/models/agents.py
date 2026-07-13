"""
Agent population for the LOB ABM (Section 2.2 of the paper).

Four core agent types, matching Section 2.2 exactly:
  - InstitutionalAgent: sole source of metaorders, splits Q into child orders.
  - HFTMarketMakerAgent: two-sided quoting + replenishment.
  - RetailAgent: small random-direction noise trades.
  - NewsAgent: rare large market orders (exogenous volatility).

Plus MomentumAgent, which only exists in the "momentum" counterfactual
scenario of Section 4.3 / Table 1 (not part of the baseline population).

WARNING: low-confidence implementation for HFTMarketMakerAgent's adaptive
spread rule and NewsAgent's trigger process — see the module docstrings
below and SIR ambiguities[0], ambiguities[1] (confidence 0.4, 0.35).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sqrt_law_abm.models.lob import LimitOrderBook, Side


@dataclass
class Metaorder:
    """A single institutional metaorder before it is split into children.

    Args:
        agent_id: Owning InstitutionalAgent's id.
        side: "buy" or "sell".
        total_size: Total size Q (Section 2.2).
        n_children: Number of child orders Nc it will be split into.
    """

    agent_id: int
    side: Side
    total_size: float
    n_children: int


@dataclass
class ChildOrder:
    """A single child order scheduled for submission at a future step."""

    agent_id: int
    side: Side
    size: float
    t_submit: int


class BaseAgent:
    """Common interface for all agent types.

    Args:
        agent_id: Unique integer id for this agent within a simulation.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def act(self, lob: LimitOrderBook, t: int, rng: np.random.Generator) -> None:
        """Perform this agent's action for step t. Subclasses must override."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.agent_id})"


class InstitutionalAgent(BaseAgent):
    """Generates and executes metaorders via child-order splitting (Sec. 2.2).

    Args:
        agent_id: Unique agent id (0..19 for the 20 baseline institutions).
        pareto_xi: Shape parameter for the Pareto metaorder-size distribution,
            drawn once per agent from Uniform[1.5, 3.5] (Sec. 2.2).
        qmin: This agent's minimum order size, log-spaced across agents from
            1 to 2000 (Sec. 2.2) so that Q/V_D spans 0.01%-10%.
        poisson_gap_lambda_ticks: Mean inter-child arrival gap in ticks
            (Sec. 2.2: lambda = 3 ticks).
        splitting_rule: One of "dirichlet", "uniform", "front_loaded".
        dirichlet_concentration: Concentration parameter for the Dirichlet
            splitting rule.
            ASSUMED (SIR confidence 0.55): the paper names "Dirichlet" as the
            default splitting rule but does not give its concentration
            parameter.
    """

    def __init__(
        self,
        agent_id: int,
        pareto_xi: float,
        qmin: float,
        poisson_gap_lambda_ticks: float = 3.0,
        splitting_rule: str = "dirichlet",
        dirichlet_concentration: float = 1.0,
        child_count_range: tuple[int, int] = (5, 50),
    ):
        super().__init__(agent_id)
        self.pareto_xi = pareto_xi
        self.qmin = qmin
        self.poisson_gap_lambda_ticks = poisson_gap_lambda_ticks
        self.splitting_rule = splitting_rule
        self.dirichlet_concentration = dirichlet_concentration
        self.child_count_range = child_count_range

        self._pending_children: list[ChildOrder] = []
        self._active_metaorder: Metaorder | None = None

    def generate_metaorder(self, rng: np.random.Generator) -> Metaorder:
        """Draw a new metaorder: total size from a Pareto distribution (Sec. 2.2)."""
        # Pareto(xi) with minimum qmin: Q = qmin * (1 + Pareto_sample)
        u = rng.uniform(0, 1)
        total_size = self.qmin * (1.0 - u) ** (-1.0 / self.pareto_xi)
        side: Side = "buy" if rng.uniform() < 0.5 else "sell"
        n_children = int(rng.integers(self.child_count_range[0], self.child_count_range[1] + 1))
        return Metaorder(self.agent_id, side, total_size, n_children)

    def split_into_children(
        self, metaorder: Metaorder, rng: np.random.Generator, t_start: int
    ) -> list[ChildOrder]:
        """Split a metaorder's total size into Nc child orders (Sec. 2.2).

        Child arrival times follow a Poisson process with mean gap
        `poisson_gap_lambda_ticks`. Child sizes follow `self.splitting_rule`:
          - "dirichlet": sizes ~ Dirichlet(concentration) * total_size
          - "uniform": all children equal size total_size / Nc
          - "front_loaded": geometrically decaying shares, largest first
            (ASSUMED interpretation, SIR ambiguities[3], confidence 0.5)
        """
        n = metaorder.n_children
        if self.splitting_rule == "dirichlet":
            alpha = np.full(n, self.dirichlet_concentration)
            shares = rng.dirichlet(alpha)
        elif self.splitting_rule == "uniform":
            shares = np.full(n, 1.0 / n)
        elif self.splitting_rule == "front_loaded":
            # ASSUMED: geometric decay, ratio 0.85, renormalized to sum to 1
            ratio = 0.85
            raw = ratio ** np.arange(n)
            shares = raw / raw.sum()
        else:
            raise ValueError(f"Unknown splitting_rule: {self.splitting_rule}")

        sizes = shares * metaorder.total_size
        gaps = rng.poisson(self.poisson_gap_lambda_ticks, size=n)
        gaps[0] = 0
        arrival_times = t_start + np.cumsum(gaps)

        children = [
            ChildOrder(self.agent_id, metaorder.side, float(sizes[i]), int(arrival_times[i]))
            for i in range(n)
        ]
        return children

    def act(self, lob: LimitOrderBook, t: int, rng: np.random.Generator) -> None:
        """Submit any child orders due at step t; start a new metaorder if idle."""
        if self._active_metaorder is None and not self._pending_children:
            metaorder = self.generate_metaorder(rng)
            self._active_metaorder = metaorder
            self._pending_children = self.split_into_children(metaorder, rng, t_start=t)

        due = [c for c in self._pending_children if c.t_submit <= t]
        for child in due:
            lob.submit_market_order(child.side, child.size, self.agent_id, t)
        self._pending_children = [c for c in self._pending_children if c.t_submit > t]

        if not self._pending_children:
            self._active_metaorder = None


class HFTMarketMakerAgent(BaseAgent):
    """Two-sided market maker providing continuous liquidity (Sec. 2.2).

    WARNING: low-confidence implementation. The paper states HFT agents
    "post two-sided limit orders at l = 50 price levels with adaptive
    spread" and "replenish with probability p = 0.9" after a fill, but does
    not give the exact adaptive-spread formula. We implement a simple
    volatility-scaled spread (widens with recent realized volatility) as the
    most standard market-making heuristic; see SIR ambiguities[0]
    (confidence 0.4) and config.yaml `model.hft.spread_rule`.

    Args:
        agent_id: Unique agent id.
        quote_levels: Number of price levels quoted on each side (Sec. 2.2: 50).
        replenish_prob: Probability of reposting after a fill (Sec. 2.2: 0.9).
        base_spread_ticks: ASSUMED baseline half-spread in ticks.
        spread_vol_sensitivity: ASSUMED sensitivity of spread to recent volatility.
        quote_size: Size posted at each level.
    """

    def __init__(
        self,
        agent_id: int,
        quote_levels: int = 50,
        replenish_prob: float = 0.9,
        base_spread_ticks: float = 4.0,
        spread_vol_sensitivity: float = 2.0,
        quote_size: float = 50.0,
        depth_profile_power: float = 1.5,
    ):
        super().__init__(agent_id)
        self.quote_levels = quote_levels
        self.replenish_prob = replenish_prob
        self.base_spread_ticks = base_spread_ticks
        self.spread_vol_sensitivity = spread_vol_sensitivity
        self.quote_size = quote_size
        self.depth_profile_power = depth_profile_power
        self._recent_prices: list[float] = []
        self._our_order_ids: list[int] = []

    def _current_spread_ticks(self, lob: LimitOrderBook) -> float:
        """ASSUMED volatility-scaled spread rule (see class docstring)."""
        if len(self._recent_prices) < 10:
            return self.base_spread_ticks
        rets = np.diff(np.log(np.array(self._recent_prices[-50:]) + 1e-12))
        vol = np.std(rets) if len(rets) > 1 else 0.0
        return self.base_spread_ticks * (1.0 + self.spread_vol_sensitivity * vol * 100)

    def quote(self, lob: LimitOrderBook, t: int, rng: np.random.Generator) -> None:
        """(Re)post two-sided quotes if this agent has no resting orders.

        FIX (post-generation, found via user testing): quote size per level now
        grows as (level+1)^depth_profile_power instead of being flat across all
        50 levels. A flat profile gives a linear cumulative depth V(q) ~ q
        (gamma=0), which pushed the no-splitting-ablation delta toward 1
        instead of the paper's reported gamma in [1.5, 3.6] ("concentrated
        near-best-quote liquidity" -- thin right at the touch, growing quickly
        further out). depth_profile_power=1.5 targets the low end of that
        range; see config.yaml `model.hft.depth_profile_power` (ASSUMED, not
        given numerically in the paper -- tune this to match Figure 8's
        reported gamma if you have a live estimate from estimate_depth_profile_gamma).
        """
        for oid in self._our_order_ids:
            lob.cancel_order(oid)
        self._our_order_ids = []

        mid = lob.mid_price()
        self._recent_prices.append(mid)
        spread = self._current_spread_ticks(lob) * lob.tick_size

        for level in range(self.quote_levels):
            bid_price = mid - spread / 2 - level * lob.tick_size
            ask_price = mid + spread / 2 + level * lob.tick_size
            level_size = self.quote_size * (level + 1) ** self.depth_profile_power
            bid_id = lob.submit_limit_order("buy", bid_price, level_size, self.agent_id, t)
            ask_id = lob.submit_limit_order("sell", ask_price, level_size, self.agent_id, t)
            self._our_order_ids.extend([bid_id, ask_id])

    def act(self, lob: LimitOrderBook, t: int, rng: np.random.Generator) -> None:
        self.quote(lob, t, rng)

    def on_fill(self, lob: LimitOrderBook, t: int, rng: np.random.Generator) -> None:
        """Replenish depleted levels with probability replenish_prob (Sec. 2.2)."""
        if rng.uniform() < self.replenish_prob:
            self.quote(lob, t, rng)


class RetailAgent(BaseAgent):
    """Noise trader submitting small, randomly-directed market orders (Sec. 2.2).

    Args:
        agent_id: Unique agent id.
        size_min: Minimum order size (Sec. 2.2: 1).
        size_max: Maximum order size (Sec. 2.2: 10).
    """

    def __init__(self, agent_id: int, size_min: float = 1.0, size_max: float = 10.0):
        super().__init__(agent_id)
        self.size_min = size_min
        self.size_max = size_max

    def act(self, lob: LimitOrderBook, t: int, rng: np.random.Generator) -> None:
        side: Side = "buy" if rng.uniform() < 0.5 else "sell"
        size = rng.uniform(self.size_min, self.size_max)
        lob.submit_market_order(side, size, self.agent_id, t)


class NewsAgent(BaseAgent):
    """Occasionally injects large market orders to generate volatility spikes (Sec. 2.2).

    WARNING: low-confidence implementation. Trigger frequency and size are
    not quantified in the paper beyond "occasionally injects large market
    orders" whose "purpose is to generate exogenous volatility spikes,
    without which the simulated return distribution would lack the heavy
    tails seen in real markets." See SIR ambiguities[1] (confidence 0.35).

    Args:
        agent_id: Unique agent id.
        trigger_rate: Per-step probability of firing (ASSUMED).
        size_multiplier: Size relative to a typical retail order (ASSUMED).
    """

    def __init__(self, agent_id: int, trigger_rate: float = 0.001, size_multiplier: float = 20.0):
        super().__init__(agent_id)
        self.trigger_rate = trigger_rate
        self.size_multiplier = size_multiplier

    def act(self, lob: LimitOrderBook, t: int, rng: np.random.Generator) -> None:
        if rng.uniform() < self.trigger_rate:
            side: Side = "buy" if rng.uniform() < 0.5 else "sell"
            size = self.size_multiplier * rng.uniform(1.0, 10.0)
            lob.submit_market_order(side, size, self.agent_id, t)


class MomentumAgent(BaseAgent):
    """Trend-following agent, active only in the "momentum" ablation scenario
    (Table 1 / Section 4.3). NOT part of the baseline agent population.

    ASSUMED (SIR ambiguities[2], confidence 0.4): the paper names a
    "Momentum" perturbation scenario but does not specify the lookback
    window or trading rule; we use a standard sign-of-recent-return rule.

    Args:
        agent_id: Unique agent id.
        lookback_ticks: Number of past ticks used to compute the recent return.
        size: Order size when triggered.
    """

    def __init__(self, agent_id: int, lookback_ticks: int = 30, size: float = 5.0):
        super().__init__(agent_id)
        self.lookback_ticks = lookback_ticks
        self.size = size

    def act(
        self,
        lob: LimitOrderBook,
        price_history: list[float],
        t: int,
        rng: np.random.Generator,
    ) -> None:
        if len(price_history) <= self.lookback_ticks:
            return
        recent_return = price_history[-1] - price_history[-1 - self.lookback_ticks]
        if recent_return == 0:
            return
        side: Side = "buy" if recent_return > 0 else "sell"
        lob.submit_market_order(side, self.size, self.agent_id, t)
