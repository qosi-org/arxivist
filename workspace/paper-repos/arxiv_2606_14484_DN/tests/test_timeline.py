"""Unit tests for quantum_horizon.timeline (physics/survey estimators, forecast blend)."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_horizon.timeline import PhysicsBasedEstimator, SurveyBasedEstimator, SystemicForecastModel


def test_physics_estimator_crossing_year_before_lag():
    est = PhysicsBasedEstimator()
    year = est.crossing_year(doubling_time=1.75, halving_time=12)
    assert 2030 < year < 2060


def test_physics_estimator_crossing_year_faster_hardware_is_earlier():
    est = PhysicsBasedEstimator()
    fast = est.crossing_year(doubling_time=1.0, halving_time=12)
    slow = est.crossing_year(doubling_time=2.5, halving_time=12)
    assert fast < slow


def test_physics_estimator_sample_shape_and_range():
    est = PhysicsBasedEstimator()
    rng = np.random.default_rng(0)
    samples = est.sample_break_years(1000, rng)
    assert samples.shape == (1000,)
    assert np.all(samples > 2026)


def test_physics_estimator_median_near_paper_mode():
    """Paper states the physics estimator's mode is near 2052."""
    est = PhysicsBasedEstimator()
    rng = np.random.default_rng(1)
    samples = est.sample_break_years(50000, rng)
    assert 2045 < np.median(samples) < 2058


def test_survey_estimator_sample_shape_and_positivity():
    est = SurveyBasedEstimator()
    rng = np.random.default_rng(0)
    samples = est.sample_break_years(1000, rng)
    assert samples.shape == (1000,)
    assert np.all(samples > est.t0)


def test_survey_estimator_invalid_mode_year_raises():
    est = SurveyBasedEstimator(mode_year=2020, t0=2026)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        est.sample_break_years(10, rng)


def test_systemic_forecast_run_returns_expected_keys():
    forecast = SystemicForecastModel(PhysicsBasedEstimator(), SurveyBasedEstimator())
    result = forecast.run(n_draws=5000, survey_weight=0.5, seed=0)
    expected_keys = {
        "survey_samples", "physics_samples", "combined_samples",
        "cdf_2035", "cdf_2040", "cdf_2050", "median", "range_80pct",
    }
    assert expected_keys == set(result.keys())


def test_systemic_forecast_cdf_monotone():
    forecast = SystemicForecastModel(PhysicsBasedEstimator(), SurveyBasedEstimator())
    result = forecast.run(n_draws=20000, survey_weight=0.5, seed=0)
    assert result["cdf_2035"] <= result["cdf_2040"] <= result["cdf_2050"]


def test_systemic_forecast_matches_paper_ballpark():
    """Sanity check against the paper's headline percentiles (Section 3.2):
    ~1-in-6 by 2035, ~30% by 2040, ~60% by 2050, median ~2046-2047.
    These are approximate (SIR confidence 0.45 on the exact distributional
    form), so we check wide tolerance bands, not exact equality.
    """
    forecast = SystemicForecastModel(PhysicsBasedEstimator(), SurveyBasedEstimator())
    result = forecast.run(n_draws=200000, survey_weight=0.5, seed=0)
    assert 0.05 < result["cdf_2035"] < 0.30
    assert 0.15 < result["cdf_2040"] < 0.45
    assert 0.45 < result["cdf_2050"] < 0.75
    assert 2040 < result["median"] < 2052


def test_sensitivity_sweep_increases_by2035_with_survey_weight():
    """Paper: 'at survey weights from 0.25 to 0.75 the by-2035 probability
    runs from about 8% to 24%' -- higher survey weight should increase
    near-term probability since the survey estimator has a nearer-term mode."""
    forecast = SystemicForecastModel(PhysicsBasedEstimator(), SurveyBasedEstimator())
    df = forecast.sensitivity_sweep([0.25, 0.5, 0.75], n_draws=50000, seed=0)
    assert len(df) == 3
    values = df.sort_values("survey_weight")["cdf_2035"].values
    assert values[0] < values[-1]
