"""
Full single-stock market simulation (Sections 2.1-2.4 of the paper).

Wires the LimitOrderBook together with the four agent types (plus the
optional MomentumAgent for the "momentum" ablation scenario) into one
discrete-event simulation: at every step one agent is chosen uniformly at
random to act (Section 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sqrt_law_abm.models.agents import (
    HFTMarketMakerAgent,
    InstitutionalAgent,
    MomentumAgent,
    NewsAgent,
    RetailAgent,
)
from sqrt_law_abm.models.lob import LimitOrderBook, Trade


@dataclass
class SimulationResult:
    """Output of one StockMarketSimulation.run() call.

    Args:
        trade_tape: All executed trades (post-warmup only).
        price_series: Mid-price at every recorded step (post-warmup).
        stock_config: The configuration this simulation was run with.
        steps_per_day: Number of ticks per trading day, for daily aggregation.
    """

    trade_tape: list[Trade]
    price_series: np.ndarray
    stock_config: dict
    steps_per_day: int

    def daily_volume_and_range(self) -> tuple[dict[int, float], dict[int, float]]:
        """Aggregate trade tape into per-day volume V_D and price range sigma_D
        (Section 3.1 normalization).

        Returns:
            (daily_volume, daily_range): dicts mapping day index -> value.
        """
        daily_volume: dict[int, float] = {}
        daily_high: dict[int, float] = {}
        daily_low: dict[int, float] = {}

        for trade in self.trade_tape:
            day = trade.t // self.steps_per_day
            daily_volume[day] = daily_volume.get(day, 0.0) + trade.size
            daily_high[day] = max(daily_high.get(day, trade.price), trade.price)
            daily_low[day] = min(daily_low.get(day, trade.price), trade.price)

        daily_range = {
            day: (daily_high[day] - daily_low[day]) for day in daily_high
        }
        return daily_volume, daily_range


class StockMarketSimulation:
    """One independent stock's discrete-event ABM simulation.

    Args:
        stock_config: Dict with keys: initial_price, n_hft, n_retail,
            pareto_xi, seed, and any scenario overrides (splitting_rule,
            hft_active, price_limits_enabled, momentum_enabled, etc.).
        model_cfg: The `model:` section of the global config.
        sim_cfg: The `simulation:` section of the global config.
    """

    def __init__(self, stock_config: dict, model_cfg: dict, sim_cfg: dict):
        self.stock_config = stock_config
        self.model_cfg = model_cfg
        self.sim_cfg = sim_cfg
        self.rng = np.random.default_rng(stock_config["seed"])

        self.lob = LimitOrderBook(
            initial_price=stock_config["initial_price"],
            tick_size_bp=model_cfg["tick_size_bp"],
        )

        self._build_agents()
        self._price_history: list[float] = []

    def _build_agents(self) -> None:
        m = self.model_cfg
        cfg = self.stock_config
        splitting_rule = cfg.get("splitting_rule", m["institutional"]["splitting_rule"])
        hft_active = cfg.get("hft_active", True)
        force_single_child = cfg.get("force_single_child", False)
        child_count_range = (
            (1, 1)
            if force_single_child
            else (m["institutional"]["child_count_min"], m["institutional"]["child_count_max"])
        )

        qmins = np.logspace(
            np.log10(m["institutional"]["qmin_log_min"]),
            np.log10(m["institutional"]["qmin_log_max"]),
            m["n_institutional"],
        )
        self.institutional_agents = [
            InstitutionalAgent(
                agent_id=i,
                pareto_xi=cfg["pareto_xi"],
                qmin=qmins[i],
                poisson_gap_lambda_ticks=m["institutional"]["poisson_gap_lambda_ticks"],
                splitting_rule=splitting_rule,
                dirichlet_concentration=m["institutional"]["dirichlet_concentration"],
                child_count_range=child_count_range,
            )
            for i in range(m["n_institutional"])
        ]

        next_id = m["n_institutional"]
        self.hft_agents: list[HFTMarketMakerAgent] = []
        if hft_active:
            n_hft = cfg.get("n_hft", m["n_hft_min"])
            for _ in range(n_hft):
                self.hft_agents.append(
                    HFTMarketMakerAgent(
                        agent_id=next_id,
                        quote_levels=m["hft"]["quote_levels"],
                        replenish_prob=m["hft"]["replenish_prob"],
                        base_spread_ticks=m["hft"]["base_spread_ticks"],
                        spread_vol_sensitivity=m["hft"]["spread_vol_sensitivity"],
                        depth_profile_power=m["hft"].get("depth_profile_power", 1.5),
                    )
                )
                next_id += 1

        n_retail = cfg.get("n_retail", m["n_retail_min"])
        self.retail_agents = [
            RetailAgent(
                agent_id=next_id + i,
                size_min=m["retail"]["order_size_min"],
                size_max=m["retail"]["order_size_max"],
            )
            for i in range(n_retail)
        ]
        next_id += n_retail

        self.news_agents = [
            NewsAgent(
                agent_id=next_id + i,
                trigger_rate=m["news"]["trigger_rate"],
                size_multiplier=m["news"]["size_multiplier"],
            )
            for i in range(m["n_news"])
        ]
        next_id += m["n_news"]

        self.momentum_agents: list[MomentumAgent] = []
        if cfg.get("momentum_enabled", False):
            for i in range(m["momentum"]["n_agents"]):
                self.momentum_agents.append(
                    MomentumAgent(
                        agent_id=next_id + i,
                        lookback_ticks=m["momentum"]["lookback_ticks"],
                    )
                )
            next_id += m["momentum"]["n_agents"]

        self._all_actionable_agents = (
            self.institutional_agents + self.retail_agents + self.news_agents
        )
        # HFT and momentum agents act via dedicated hooks below (quoting /
        # trend-following are not "one random agent per step" events in the
        # same sense; they refresh quotes each step and react to fills).

    def _apply_price_limit(self, t: int) -> bool:
        """ASSUMED price-limit mechanism (SIR ambiguities[2], confidence 0.4):
        halts trading for the step if price has moved beyond `band_pct` from
        the start-of-day price. Returns True if trading should be skipped.
        """
        cfg = self.stock_config
        if not cfg.get("price_limits_enabled", False):
            return False
        band_pct = self.model_cfg["price_limits"]["band_pct"]
        day_start_idx = (t // self.sim_cfg["steps_per_day"]) * self.sim_cfg["steps_per_day"]
        if day_start_idx >= len(self._price_history):
            return False
        day_open = self._price_history[day_start_idx]
        current = self.lob.mid_price()
        return abs(current - day_open) / day_open * 100 > band_pct

    def step(self, t: int) -> None:
        """Advance the simulation by one discrete event (Section 2.4)."""
        # HFT agents refresh quotes every step to keep the book "quasi-stable"
        # (Figure 9 shows near-constant spread/depth across steps).
        for hft in self.hft_agents:
            hft.act(self.lob, t, self.rng)

        if self._apply_price_limit(t):
            self._price_history.append(self.lob.mid_price())
            return

        if self.momentum_agents and self.rng.uniform() < 0.1:
            agent = self.momentum_agents[int(self.rng.integers(len(self.momentum_agents)))]
            agent.act(self.lob, self._price_history, t, self.rng)

        agent = self._all_actionable_agents[int(self.rng.integers(len(self._all_actionable_agents)))]
        n_trades_before = len(self.lob.trade_tape)
        agent.act(self.lob, t, self.rng)

        if len(self.lob.trade_tape) > n_trades_before and self.hft_agents:
            filled_hft = [h for h in self.hft_agents]
            for hft in filled_hft:
                hft.on_fill(self.lob, t, self.rng)

        self._price_history.append(self.lob.mid_price())

    def run(self, n_steps: int, warmup_steps: int) -> SimulationResult:
        """Run warmup + recorded steps and return the SimulationResult.

        Args:
            n_steps: Number of recorded steps after warmup (Sec. 2.4: T = 1e6).
            warmup_steps: Steps to let the LOB self-organise before recording
                trades (Sec. 2.4: 5e4).
        """
        for hft in self.hft_agents:
            hft.act(self.lob, 0, self.rng)

        for t in range(warmup_steps):
            self.step(t)

        trade_count_at_warmup_end = len(self.lob.trade_tape)
        for t in range(warmup_steps, warmup_steps + n_steps):
            self.step(t)

        post_warmup_trades = self.lob.trade_tape[trade_count_at_warmup_end:]
        # Re-index trade times relative to end of warmup for cleaner day bucketing
        rebased_trades = [
            Trade(
                t=tr.t - warmup_steps,
                price=tr.price,
                size=tr.size,
                buy_agent_id=tr.buy_agent_id,
                sell_agent_id=tr.sell_agent_id,
                aggressor_side=tr.aggressor_side,
                aggressor_agent_id=tr.aggressor_agent_id,
            )
            for tr in post_warmup_trades
        ]

        return SimulationResult(
            trade_tape=rebased_trades,
            price_series=np.array(self._price_history[warmup_steps:]),
            stock_config=self.stock_config,
            steps_per_day=self.sim_cfg["steps_per_day"],
        )
