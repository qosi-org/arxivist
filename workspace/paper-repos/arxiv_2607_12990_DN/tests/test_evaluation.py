"""Unit tests for quantum_cva.evaluation (error decomposition, metrics)."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cva.evaluation import ErrorBudget, TrajectoryMetrics


def test_error_budget_matches_paper_reported_values():
    budget = ErrorBudget()
    result = budget.compute_budget(
        cva_cont_mc=1.091, cva_tab=0.522, cva_sv=0.670, cva_ae=0.670 * (1 + 0.0000227)
    )
    assert result.eps_grid == pytest.approx(0.5215, abs=1e-3)
    assert result.eps_enc == pytest.approx(0.2835, abs=1e-3)
    assert result.alpha_n == pytest.approx(0.4785, abs=1e-3)
    assert result.beta_theta == pytest.approx(0.6141, abs=1e-3)


def test_error_budget_total_bound_is_sum_of_components():
    budget = ErrorBudget()
    result = budget.compute_budget(cva_cont_mc=1.0, cva_tab=0.5, cva_sv=0.6, cva_ae=0.65)
    expected_structural = result.eps_grid + result.alpha_n * result.eps_enc
    assert result.structural_bound == pytest.approx(expected_structural)
    assert result.total_bound == pytest.approx(expected_structural + result.beta_theta * result.eps_ae)


def test_query_cost_matches_formula():
    metrics = TrajectoryMetrics()
    stages = [(0, 256), (1, 256), (3, 512)]
    expected = 256 * 1 + 256 * 3 + 512 * 7
    assert metrics.query_cost(stages) == expected


def test_fit_scaling_exponent_recovers_known_slope():
    metrics = TrajectoryMetrics()
    n_q = np.array([100, 1000, 10000, 100000], dtype=float)
    errors = 10.0 * n_q ** (-1.0)  # ideal beta = -1
    beta, alpha, r2 = metrics.fit_scaling_exponent(n_q, errors)
    assert beta == pytest.approx(-1.0, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_median_with_bootstrap_ci_reasonable():
    metrics = TrajectoryMetrics()
    values = np.array([0.01, 0.012, 0.011, 0.013, 0.009, 0.015, 0.010])
    median, lo, hi = metrics.median_with_bootstrap_ci(values, n_bootstrap=200)
    assert lo <= median <= hi
