"""Unit tests for ews_kalman.kalman (transition matrix, TVP-Kalman filter)."""

from __future__ import annotations

import numpy as np
import pytest

from ews_kalman.kalman import TVPKalmanFilter, build_taylor_transition_matrix


def test_taylor_matrix_shape_and_diagonal():
    F = build_taylor_transition_matrix(dt=1.0 / 12, order=3)
    assert F.shape == (4, 4)
    assert np.allclose(np.diag(F), 1.0)


def test_taylor_matrix_known_entries():
    dt = 0.5
    F = build_taylor_transition_matrix(dt=dt, order=3)
    assert F[0, 1] == pytest.approx(dt)
    assert F[0, 2] == pytest.approx(dt**2 / 2)
    assert F[0, 3] == pytest.approx(dt**3 / 6)
    assert F[1, 2] == pytest.approx(dt)
    assert np.allclose(np.tril(F, k=-1), 0.0)


def test_taylor_matrix_order_generalizes():
    F = build_taylor_transition_matrix(dt=1.0, order=1)
    assert F.shape == (2, 2)
    assert np.allclose(F, np.array([[1.0, 1.0], [0.0, 1.0]]))


def test_kalman_filter_recovers_constant_loglog_elasticity():
    """If y = x^beta exactly (no noise), the loglog filter should recover
    beta almost exactly after settling."""
    rng = np.random.default_rng(0)
    N = 200
    beta_true = 0.5
    x = 100 + rng.normal(0, 5, N).cumsum() * 0.1
    x = np.abs(x) + 50  # keep positive and away from 0
    y = x**beta_true

    kf = TVPKalmanFilter(R=1e-6, Q_diag=(1e-8, 1e-9, 1e-10, 1e-11), dt=1.0, mode="loglog")
    result = kf.estimate_beta(x, y)
    # Allow the first ~20 steps for filter settling
    assert np.mean(np.abs(result["beta"][20:] - beta_true)) < 0.05


def test_kalman_filter_linear_mode_recovers_constant_coefficient():
    """If y = beta*x + small noise, the linear-mode filter should recover
    beta after settling."""
    rng = np.random.default_rng(1)
    N = 200
    beta_true = 0.6
    x = rng.normal(0, 1, N).cumsum() * 0.1 + 10
    y = beta_true * x + rng.normal(0, 0.01, N)

    kf = TVPKalmanFilter(R=1e-3, Q_diag=(1e-6, 1e-7, 1e-8, 1e-9), dt=1.0, mode="linear")
    result = kf.estimate_beta(x, y)
    assert np.mean(np.abs(result["beta"][50:] - beta_true)) < 0.1


def test_kalman_filter_tracks_step_change_in_linear_mode():
    rng = np.random.default_rng(2)
    N = 300
    tipping = 200
    x = rng.normal(0, 1, N).cumsum() * 0.1 + 10
    beta_true = np.where(np.arange(N) < tipping, 0.8, 0.2)
    y = beta_true * x + rng.normal(0, 0.05, N)

    kf = TVPKalmanFilter(R=1e-3, Q_diag=(1e-4, 1e-5, 1e-6, 1e-7), dt=1.0, mode="linear")
    result = kf.estimate_beta(x, y)
    beta_est = result["beta"]
    # Before the step (excluding warm-up), estimate should be closer to 0.8 than 0.2
    assert abs(np.mean(beta_est[50:190]) - 0.8) < abs(np.mean(beta_est[50:190]) - 0.2)
    # After the step, estimate should be closer to 0.2 than 0.8
    assert abs(np.mean(beta_est[250:]) - 0.2) < abs(np.mean(beta_est[250:]) - 0.8)


def test_kalman_filter_invalid_mode_raises():
    with pytest.raises(ValueError):
        TVPKalmanFilter(mode="bogus")


def test_kalman_smooth_matches_filter_at_last_timestep():
    rng = np.random.default_rng(3)
    N = 50
    x = np.abs(rng.normal(100, 5, N))
    y = x**0.4

    kf = TVPKalmanFilter(mode="loglog")
    x_filt, P_filt, x_pred, P_pred = kf.filter(np.log(x), np.log(y))
    x_smooth, _ = kf.smooth(x_filt, P_filt, x_pred, P_pred)
    assert np.allclose(x_smooth[-1], x_filt[-1])
