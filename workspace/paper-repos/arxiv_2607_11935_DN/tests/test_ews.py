"""Unit tests for ews_kalman.ews (classical EWS, lead-lag analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from ews_kalman.ews import ClassicalEWS, LeadLagAnalyzer


def test_rolling_ar1_output_length():
    ews = ClassicalEWS()
    x = np.arange(100, dtype=float)
    ar1 = ews.rolling_ar1(x, window=24)
    assert len(ar1) == 100 - 24 + 1


def test_rolling_ar1_high_for_smooth_trend():
    ews = ClassicalEWS()
    x = np.linspace(0, 100, 60)  # perfectly linear trend
    ar1 = ews.rolling_ar1(x, window=24)
    assert np.mean(ar1) > 0.9  # near-perfect autocorrelation for a linear ramp


def test_rolling_ar1_near_zero_for_white_noise():
    rng = np.random.default_rng(0)
    ews = ClassicalEWS()
    x = rng.normal(0, 1, 500)
    ar1 = ews.rolling_ar1(x, window=24)
    assert abs(np.mean(ar1)) < 0.3  # white noise: AR1 should hover near 0


def test_rolling_variance_output_length_and_values():
    ews = ClassicalEWS()
    x = np.concatenate([np.zeros(30), np.ones(30) * 10])
    var = ews.rolling_variance(x, window=24)
    assert len(var) == len(x) - 24 + 1
    assert var[0] == pytest.approx(0.0, abs=1e-9)  # first window is all zeros


def test_permutation_entropy_bounds():
    rng = np.random.default_rng(1)
    ews = ClassicalEWS()
    x = rng.normal(0, 1, 200)
    pe = ews.rolling_permutation_entropy(x, embedding_dim=3, window=36)
    assert np.all(pe >= 0.0) and np.all(pe <= 1.0 + 1e-9)


def test_permutation_entropy_higher_for_noise_than_monotone():
    rng = np.random.default_rng(2)
    ews = ClassicalEWS()
    noise = rng.normal(0, 1, 200)
    monotone = np.arange(200, dtype=float)
    pe_noise = np.mean(ews.rolling_permutation_entropy(noise, window=36))
    pe_monotone = np.mean(ews.rolling_permutation_entropy(monotone, window=36))
    assert pe_noise > pe_monotone


def test_mutual_information_output_length_and_nonnegative():
    rng = np.random.default_rng(3)
    ews = ClassicalEWS()
    x = rng.normal(0, 1, 150)
    y = 2 * x + rng.normal(0, 0.1, 150)
    mi = ews.rolling_mutual_information(x, y, window=36)
    assert len(mi) == 150 - 36 + 1
    assert np.all(mi >= -1e-9)


def test_mutual_information_higher_for_correlated_than_independent():
    rng = np.random.default_rng(4)
    ews = ClassicalEWS()
    x = rng.normal(0, 1, 200)
    y_correlated = 3 * x + rng.normal(0, 0.05, 200)
    y_independent = rng.normal(0, 1, 200)
    mi_corr = np.mean(ews.rolling_mutual_information(x, y_correlated, window=36))
    mi_indep = np.mean(ews.rolling_mutual_information(x, y_independent, window=36))
    assert mi_corr > mi_indep


def test_optimal_lag_detects_known_shift():
    analyzer = LeadLagAnalyzer()
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, 200).cumsum()
    shift = 10
    a = base[: 200 - shift]
    b = base[shift:]
    lag, corr = analyzer.optimal_lag(a, b, max_lag=20)
    assert lag == pytest.approx(-shift, abs=2)
    assert corr > 0.5


def test_pearson_orthogonality_near_zero_for_independent_signals():
    analyzer = LeadLagAnalyzer()
    rng = np.random.default_rng(6)
    a = rng.normal(0, 1, 300)
    b = rng.normal(0, 1, 300)
    r, p = analyzer.pearson_orthogonality(a, b)
    assert abs(r) < 0.2


def test_first_significant_deviation_detects_shift():
    analyzer = LeadLagAnalyzer()
    signal = np.concatenate([np.zeros(50), np.full(50, 10.0)])
    idx = analyzer.first_significant_deviation(signal, baseline_mean=0.0, baseline_std=1.0, z_threshold=2.0)
    assert idx == 50


def test_first_significant_deviation_none_when_never_crosses():
    analyzer = LeadLagAnalyzer()
    signal = np.zeros(50)
    idx = analyzer.first_significant_deviation(signal, baseline_mean=0.0, baseline_std=1.0, z_threshold=2.0)
    assert idx is None


def test_simulation_lead_time_positive_before_tipping():
    analyzer = LeadLagAnalyzer()
    # Deterministic tiny alternating pattern gives a well-defined nonzero
    # baseline std without any chance of a spurious >2-sigma excursion
    # (unlike random noise, which occasionally exceeds threshold by chance).
    baseline_pattern = np.tile([0.01, -0.01], 40)
    signal = np.concatenate([baseline_pattern, np.full(20, 10.0)])
    lead = analyzer.simulation_lead_time(signal, tipping_index=100, baseline_window=(0, 50))
    assert lead == 20  # detected at index 80, tipping at 100 -> lead = 20
