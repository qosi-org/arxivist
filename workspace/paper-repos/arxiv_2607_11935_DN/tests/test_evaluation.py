"""Unit tests for ews_kalman.evaluation (region stats, simulation validation)."""

from __future__ import annotations

import numpy as np

from ews_kalman.data import AIRSDataLoader
from ews_kalman.ews import ClassicalEWS
from ews_kalman.evaluation import RegionSummaryComputer, SimulationValidator
from ews_kalman.kalman import TVPKalmanFilter


def test_count_regime_transitions_counts_sign_changes():
    computer = RegionSummaryComputer()
    beta_dp = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
    assert computer.count_regime_transitions(beta_dp) == 3


def test_count_regime_transitions_zero_for_constant_sign():
    computer = RegionSummaryComputer()
    beta_dp = np.ones(20)
    assert computer.count_regime_transitions(beta_dp) == 0


def test_compute_table1_row_matches_tropics_ballpark():
    """End-to-end sanity check: synthetic Tropics region should produce a
    Table-1-style row with |beta| near the paper's reported ~0.49 and low
    sigma_beta (stable coupling)."""
    loader = AIRSDataLoader()
    region = loader.load_region("tropics", data_dir="/nonexistent/path", seed=0)

    kf = TVPKalmanFilter()
    beta_result = kf.estimate_beta(region["T"], region["q"])

    ews = ClassicalEWS()
    ar1_T = ews.rolling_ar1(region["T"], window=24)
    mi = ews.rolling_mutual_information(region["T"], region["q"], window=36)

    computer = RegionSummaryComputer()
    row = computer.compute_table1_row(
        beta_result["beta"], ar1_T, mi, beta_result["beta_double_prime"]
    )

    assert row["N"] == 284
    assert 0.3 < row["abs_beta_mean"] < 0.7  # paper reports 0.492
    assert row["sigma_beta"] < 0.2  # paper reports 0.035 (low volatility)


def test_compute_table2_row_structure():
    computer = RegionSummaryComputer()
    rng = np.random.default_rng(0)
    beta_derivatives = {
        "beta": rng.normal(0, 1, 100),
        "beta_prime": rng.normal(0, 1, 100),
    }
    classical_signals = {
        "AR1_T": rng.normal(0, 1, 100),
        "MI": rng.normal(0, 1, 100),
    }
    result = computer.compute_table2_row(beta_derivatives, classical_signals, max_lag=10)
    assert set(result.keys()) == {"beta", "beta_prime"}
    assert set(result["beta"].keys()) == {"AR1_T", "MI"}
    assert "lag" in result["beta"]["AR1_T"]
    assert result["beta"]["AR1_T"]["direction"] in ("lead", "lag", "coincident")


def test_simulation_validator_returns_six_systems():
    validator = SimulationValidator()
    results = validator.validate_all_systems(seed=0)
    assert len(results) == 6
    names = {r["simulation"] for r in results}
    assert names == {
        "Fold bifurcation", "Beta step change", "Beta linear decay",
        "Logistic map", "Stommel AMOC", "Critical slowing down",
    }


def test_simulation_validator_winner_field_valid():
    validator = SimulationValidator()
    results = validator.validate_all_systems(seed=0)
    for r in results:
        assert r["winner"] in ("beta", "AR1", "tie", "neither")


def test_simulation_validator_stommel_amoc_beta_wins_or_ties():
    """The paper reports beta clearly wins on Stommel AMOC (a coupling-
    degradation system); check our reimplementation agrees qualitatively."""
    validator = SimulationValidator()
    results = validator.validate_all_systems(seed=0)
    amoc_result = next(r for r in results if r["simulation"] == "Stommel AMOC")
    assert amoc_result["winner"] in ("beta", "tie")


def test_simulation_validator_reproducible_with_same_seed():
    validator = SimulationValidator()
    r1 = validator.validate_all_systems(seed=7)
    r2 = validator.validate_all_systems(seed=7)
    assert r1 == r2
