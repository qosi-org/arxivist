"""Unit tests for quantum_cva.finance (classical CVA benchmark, grid encoding)."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cva.finance import (
    BlackScholesPricer,
    CDSBootstrapper,
    CVAEstimator,
    FiniteGridBuilder,
    Instrument,
    MultiAssetGBMSimulator,
)


def test_black_scholes_call_put_parity():
    pricer = BlackScholesPricer()
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.02, 0.01, 0.2
    call = pricer.call_price(S, K, T, r, q, sigma)
    put = pricer.put_price(S, K, T, r, q, sigma)
    forward = pricer.forward_price(S, K, T, r, q)
    # Put-call parity: C - P = S*e^{-qT} - K*e^{-rT} = forward value
    assert (call - put) == pytest.approx(forward, abs=1e-6)


def test_black_scholes_call_price_positive():
    pricer = BlackScholesPricer()
    price = pricer.call_price(100, 90, 0.5, 0.02, 0.0, 0.25)
    assert price > 0


def test_multi_asset_gbm_simulator_shapes():
    spots = {"A": 100.0, "B": 200.0}
    dividends = {"A": 0.01, "B": 0.02}
    vols = {"A": np.full(4, 0.2), "B": np.full(4, 0.15)}
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    sim = MultiAssetGBMSimulator(spots, dividends, risk_free_rate=0.025, volatilities=vols, correlation_matrix=corr)
    dates = np.array([0.125, 0.25, 0.375, 0.5])
    paths = sim.simulate_paths(n_paths=1000, monitoring_dates=dates, seed=123)
    assert paths.shape == (1000, 4, 2)
    assert np.all(paths > 0)


def test_multi_asset_gbm_correlation_regularisation_handles_bad_matrix():
    spots = {"A": 100.0, "B": 200.0}
    dividends = {"A": 0.01, "B": 0.02}
    vols = {"A": np.full(2, 0.2), "B": np.full(2, 0.15)}
    # Slightly invalid correlation matrix (not PSD)
    corr = np.array([[1.0, 1.5], [1.5, 1.0]])
    sim = MultiAssetGBMSimulator(spots, dividends, 0.02, vols, corr)
    assert sim._cholesky.shape == (2, 2)


def test_cds_bootstrapper_survival_curve_monotone_decreasing():
    bootstrapper = CDSBootstrapper()
    tenors = np.array([1.0, 2.0, 5.0])
    spreads = np.array([0.01, 0.012, 0.015])
    discount_fn = lambda t: np.exp(-0.02 * t)
    survival_fn = bootstrapper.bootstrap_survival_curve(tenors, spreads, recovery_cds=0.4, discount_fn=discount_fn)
    p1 = survival_fn(1.0)
    p2 = survival_fn(2.0)
    p5 = survival_fn(5.0)
    assert 1.0 >= p1 >= p2 >= p5 >= 0.0


def test_finite_grid_builder_truncate_domain():
    pricer = BlackScholesPricer()
    builder = FiniteGridBuilder(pricer)
    lower, upper = builder.truncate_domain(mu=100.0, sigma=20.0, n_std=3.0)
    assert lower == pytest.approx(40.0)
    assert upper == pytest.approx(160.0)


def test_finite_grid_builder_truncate_domain_clips_at_zero():
    pricer = BlackScholesPricer()
    builder = FiniteGridBuilder(pricer)
    lower, upper = builder.truncate_domain(mu=10.0, sigma=20.0, n_std=3.0)
    assert lower == 0.0


def test_finite_grid_builder_bin_edges_count():
    pricer = BlackScholesPricer()
    builder = FiniteGridBuilder(pricer)
    edges = builder.bin_edges(0.0, 100.0, n_bins=4)
    assert len(edges) == 5


def test_probability_tensor_sums_to_one():
    pricer = BlackScholesPricer()
    builder = FiniteGridBuilder(pricer)
    n_paths = 2000
    dates = np.array([0.25, 0.5])
    rng = np.random.default_rng(0)
    paths = np.stack(
        [rng.normal(100, 10, (n_paths, 1)) for _ in range(2)], axis=1
    ).reshape(n_paths, 2, 1)
    edges = [builder.bin_edges(50, 150, 4)]
    tensor = builder.build_probability_tensor(paths, edges, dates)
    assert tensor.shape == (2, 4)
    assert np.isclose(tensor.sum(), 1.0, atol=1e-6)


def test_rescale_constants_positive():
    pricer = BlackScholesPricer()
    builder = FiniteGridBuilder(pricer)
    exposure = np.array([[1.0, 2.0], [3.0, 0.5]])
    discount = np.array([0.99, 0.98])
    default_incr = np.array([1e-4, 2e-4])
    c_v, c_p, c_q = builder.rescale_constants(exposure, discount, default_incr)
    assert c_v >= exposure.max()
    assert c_p >= discount.max()
    assert c_q >= default_incr.max()
