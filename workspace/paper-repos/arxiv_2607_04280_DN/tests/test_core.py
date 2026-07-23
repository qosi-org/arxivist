"""Unit tests for the LOB engine, agents, and evaluation metrics.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqrt_law_abm.data.transforms import MetaorderReconstructor
from sqrt_law_abm.evaluation.metrics import (
    ImpactCurveFitter,
    TailExponentEstimator,
    TheoryPredictors,
    estimate_depth_profile_gamma,
)
from sqrt_law_abm.models.agents import InstitutionalAgent
from sqrt_law_abm.models.lob import LimitOrderBook
from sqrt_law_abm.training.losses import RelativeLeastSquaresFit


class TestLimitOrderBook:
    def test_tick_size_is_one_bp_of_initial_price(self):
        lob = LimitOrderBook(initial_price=10_000, tick_size_bp=1.0)
        assert lob.tick_size == pytest.approx(1.0)

    def test_limit_order_then_market_order_executes(self):
        lob = LimitOrderBook(initial_price=100.0)
        lob.submit_limit_order("sell", 100.05, size=10, agent_id=1, t=0)
        trades = lob.submit_market_order("buy", size=5, agent_id=2, t=1)
        assert len(trades) == 1
        assert trades[0].size == 5
        assert trades[0].buy_agent_id == 2
        assert trades[0].sell_agent_id == 1

    def test_market_order_walks_multiple_levels(self):
        lob = LimitOrderBook(initial_price=100.0)
        lob.submit_limit_order("sell", 100.01, size=5, agent_id=1, t=0)
        lob.submit_limit_order("sell", 100.02, size=5, agent_id=2, t=0)
        trades = lob.submit_market_order("buy", size=8, agent_id=3, t=1)
        assert sum(tr.size for tr in trades) == pytest.approx(8)
        assert len(trades) == 2  # fills the first level fully, second partially

    def test_cancel_order_removes_it(self):
        lob = LimitOrderBook(initial_price=100.0)
        oid = lob.submit_limit_order("buy", 99.99, size=5, agent_id=1, t=0)
        assert lob.cancel_order(oid) is True
        assert lob.n_resting_orders() == 0
        assert lob.cancel_order(oid) is False  # already gone

    def test_market_order_on_empty_book_returns_no_trades(self):
        lob = LimitOrderBook(initial_price=100.0)
        trades = lob.submit_market_order("buy", size=5, agent_id=1, t=0)
        assert trades == []

    def test_depth_profile_is_cumulative(self):
        lob = LimitOrderBook(initial_price=100.0)
        lob.submit_limit_order("sell", 100.01, size=5, agent_id=1, t=0)
        lob.submit_limit_order("sell", 100.02, size=5, agent_id=1, t=0)
        profile = lob.depth_profile("sell", n_levels=2)
        assert profile[0] == pytest.approx(5)
        assert profile[1] == pytest.approx(10)


class TestInstitutionalAgent:
    def test_generate_metaorder_size_at_least_qmin(self):
        agent = InstitutionalAgent(agent_id=0, pareto_xi=2.0, qmin=10.0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            mo = agent.generate_metaorder(rng)
            assert mo.total_size >= 10.0
            assert mo.side in ("buy", "sell")

    def test_split_children_sum_to_total_size(self):
        agent = InstitutionalAgent(agent_id=0, pareto_xi=2.0, qmin=10.0, splitting_rule="dirichlet")
        rng = np.random.default_rng(1)
        mo = agent.generate_metaorder(rng)
        children = agent.split_into_children(mo, rng, t_start=0)
        assert sum(c.size for c in children) == pytest.approx(mo.total_size)
        assert len(children) == mo.n_children

    def test_uniform_split_gives_equal_sizes(self):
        agent = InstitutionalAgent(agent_id=0, pareto_xi=2.0, qmin=10.0, splitting_rule="uniform")
        rng = np.random.default_rng(2)
        mo = agent.generate_metaorder(rng)
        children = agent.split_into_children(mo, rng, t_start=0)
        sizes = [c.size for c in children]
        assert max(sizes) == pytest.approx(min(sizes))

    def test_single_child_when_range_is_one_one(self):
        agent = InstitutionalAgent(
            agent_id=0, pareto_xi=2.0, qmin=10.0, splitting_rule="uniform", child_count_range=(1, 1)
        )
        rng = np.random.default_rng(3)
        mo = agent.generate_metaorder(rng)
        children = agent.split_into_children(mo, rng, t_start=0)
        assert len(children) == 1
        assert children[0].size == pytest.approx(mo.total_size)


class TestRelativeLeastSquaresFit:
    def test_recovers_known_power_law(self):
        rng = np.random.default_rng(0)
        x = np.logspace(-2, 1, 50)
        true_c, true_delta = 1.2, 0.55
        y = true_c * x**true_delta * (1 + rng.normal(0, 0.01, size=len(x)))
        c, delta = RelativeLeastSquaresFit().fit(x, y)
        assert delta == pytest.approx(true_delta, abs=0.05)
        assert c == pytest.approx(true_c, rel=0.2)

    def test_raises_on_too_few_points(self):
        with pytest.raises(ValueError):
            RelativeLeastSquaresFit().fit(np.array([1.0]), np.array([1.0]))


class TestTheoryPredictors:
    def test_ggps_delta(self):
        assert TheoryPredictors().ggps_delta(2.1) == pytest.approx(1.1)

    def test_fglw_delta(self):
        assert TheoryPredictors().fglw_delta(2.8) == pytest.approx(1.8)

    def test_lob_walking_delta(self):
        assert TheoryPredictors().lob_walking_delta(1.0) == pytest.approx(0.5)


class TestTailExponentEstimator:
    def test_hill_estimator_on_known_pareto(self):
        rng = np.random.default_rng(0)
        true_alpha = 2.5
        # Pareto(alpha) samples via inverse CDF, xm=1
        u = rng.uniform(0, 1, size=20_000)
        samples = (1 - u) ** (-1.0 / true_alpha)
        est = TailExponentEstimator().hill_estimator(samples, tail_fraction=0.05)
        assert est == pytest.approx(true_alpha, rel=0.25)

    def test_raises_on_too_few_values(self):
        with pytest.raises(ValueError):
            TailExponentEstimator().hill_estimator(np.array([1.0, 2.0]))


class TestEstimateDepthProfileGamma:
    def test_recovers_known_gamma(self):
        gamma_true = 1.0
        q = np.arange(1, 51)
        v = q ** (1 + gamma_true)
        est = estimate_depth_profile_gamma(v)
        assert est == pytest.approx(gamma_true, abs=0.05)


class TestMetaorderReconstructor:
    def test_groups_consecutive_same_direction_trades(self):
        from sqrt_law_abm.models.lob import Trade

        trades = [
            Trade(t=0, price=100.0, size=5, buy_agent_id=1, sell_agent_id=2, aggressor_side="buy", aggressor_agent_id=1),
            Trade(t=2, price=100.1, size=5, buy_agent_id=1, sell_agent_id=2, aggressor_side="buy", aggressor_agent_id=1),
            Trade(t=50, price=99.9, size=5, buy_agent_id=1, sell_agent_id=2, aggressor_side="sell", aggressor_agent_id=1),
        ]
        recon = MetaorderReconstructor()
        metaorders = recon.reconstruct(trades, delta_t_ticks=10, steps_per_day=5000)
        assert len(metaorders) == 2  # buy group + sell group (direction flip splits them)
        assert metaorders[0].total_size == pytest.approx(10)
        assert metaorders[1].total_size == pytest.approx(5)

    def test_gap_larger_than_threshold_splits_group(self):
        from sqrt_law_abm.models.lob import Trade

        trades = [
            Trade(t=0, price=100.0, size=5, buy_agent_id=1, sell_agent_id=2, aggressor_side="buy", aggressor_agent_id=1),
            Trade(t=20, price=100.1, size=5, buy_agent_id=1, sell_agent_id=2, aggressor_side="buy", aggressor_agent_id=1),
        ]
        recon = MetaorderReconstructor()
        metaorders = recon.reconstruct(trades, delta_t_ticks=10, steps_per_day=5000)
        assert len(metaorders) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
