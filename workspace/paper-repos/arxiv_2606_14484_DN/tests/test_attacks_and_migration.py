"""Unit tests for quantum_horizon.attacks.mempool_race and quantum_horizon.migration.migration_race."""

from __future__ import annotations

import pytest

from quantum_horizon.attacks import MempoolRaceModel
from quantum_horizon.migration import MigrationRaceModel


def test_mempool_best_case_matches_paper():
    """Paper: 9-min derivation, zero propagation, single confirmation -> ~41%."""
    mp = MempoolRaceModel()
    result = mp.snipe_success_probability(9, 0, 1)
    assert result == pytest.approx(0.41, abs=0.02)


def test_mempool_realistic_matches_paper():
    """Paper: realistic estimate ~30% for a fast-clock CRQC."""
    mp = MempoolRaceModel()
    result = mp.snipe_success_probability(10.5, 2.0, 1)
    assert result == pytest.approx(0.30, abs=0.05)


def test_mempool_more_derivation_time_lowers_success():
    mp = MempoolRaceModel()
    fast = mp.snipe_success_probability(9, 0, 1)
    slow = mp.snipe_success_probability(12, 0, 1)
    assert slow < fast


def test_mempool_more_confirmations_lowers_success():
    mp = MempoolRaceModel()
    one_conf = mp.snipe_success_probability(9, 0, 1)
    two_conf = mp.snipe_success_probability(9, 0, 2)
    assert two_conf < one_conf


def test_mempool_fast_clock_feasible():
    mp = MempoolRaceModel()
    assert mp.is_feasible("superconducting") is True
    assert mp.is_feasible("photonic") is True


def test_mempool_slow_clock_infeasible():
    """Paper: 'impossible for a slow-clock machine (trapped-ion or neutral-atom)'."""
    mp = MempoolRaceModel()
    assert mp.is_feasible("trapped-ion") is False
    assert mp.is_feasible("neutral-atom") is False


def test_mempool_unknown_clock_type_raises():
    mp = MempoolRaceModel()
    with pytest.raises(ValueError):
        mp.is_feasible("unobtanium")


def test_migration_race_mosca_inequality_basic():
    mr = MigrationRaceModel()
    assert mr.is_at_risk(start_year=2033, migration_duration_years=5, crqc_arrival_year=2035) is True
    assert mr.is_at_risk(start_year=2026, migration_duration_years=4, crqc_arrival_year=2035) is False


def test_migration_race_prompt_start_beats_aggressive_crqc():
    """Paper: 'a prompt start in 2026, migration finishes around 2029-2031
    and beats even an optimistic 2035 CRQC.'"""
    mr = MigrationRaceModel()
    assert mr.is_at_risk(start_year=2026, migration_duration_years=4, crqc_arrival_year=2035) is False


def test_migration_race_only_severe_delay_vs_aggressive_is_at_risk():
    """Paper: 'the only at-risk case in the entire sweep is a severely
    delayed start... running against an early machine.'"""
    mr = MigrationRaceModel()
    scenarios = {
        "prompt": {"start_year": 2026, "migration_duration_years": 4},
        "delayed": {"start_year": 2030, "migration_duration_years": 5},
        "severe_delay": {"start_year": 2033, "migration_duration_years": 5},
    }
    crqc_estimates = {"aggressive": 2035, "survey_median": 2040, "conservative": 2050}
    df = mr.run_scenarios(scenarios, crqc_estimates)
    at_risk_rows = df[df["at_risk"]]
    assert len(at_risk_rows) == 1
    assert at_risk_rows.iloc[0]["scenario"] == "severe_delay"
    assert at_risk_rows.iloc[0]["crqc_estimate_name"] == "aggressive"


def test_migration_race_dataframe_shape():
    mr = MigrationRaceModel()
    scenarios = {"prompt": {"start_year": 2026, "migration_duration_years": 4}}
    crqc_estimates = {"aggressive": 2035, "conservative": 2050}
    df = mr.run_scenarios(scenarios, crqc_estimates)
    assert len(df) == 2
    assert set(df.columns) == {
        "scenario", "start_year", "migration_duration_years", "finish_year",
        "crqc_estimate_name", "crqc_arrival_year", "at_risk",
    }
