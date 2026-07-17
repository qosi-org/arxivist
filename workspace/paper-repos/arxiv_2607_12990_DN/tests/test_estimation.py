"""Unit tests for quantum_cva.estimation (CABIQAE, BIQAE, BAE, DCS)."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cva.estimation import BAE, BIQAE, CABIQAE, ContrastCalibrator, DirectCircuitSampling


def make_ideal_executor(a_true: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta_true = np.arcsin(np.sqrt(a_true))

    def executor(k: int, n_shots: int):
        K = 2 * k + 1
        p = np.sin(K * theta_true) ** 2
        return int(rng.binomial(n_shots, p)), n_shots

    return executor


def test_contrast_model_decays_with_k():
    est = CABIQAE(c0=1.0, tau_c=10.0, b=0.5)
    c0_val = est.contrast_model(0)
    c5_val = est.contrast_model(5)
    assert c5_val < c0_val
    assert c0_val <= 1.0 + 1e-9


def test_p_obs_ideal_limit_matches_sin_squared():
    est = CABIQAE(c0=1.0, tau_c=1e12, b=0.5)  # tau_c -> inf, ideal limit
    theta = np.array([0.3, 0.7, 1.1])
    p = est.p_obs(theta, k=3)
    K = 7
    expected = np.sin(K * theta) ** 2
    assert np.allclose(p, expected, atol=1e-6)


def test_bayesian_update_matches_conjugate_formula():
    est = CABIQAE()
    alpha_post, beta_post = est.bayesian_update(0.5, 0.5, n_shots=100, n_success=60)
    assert alpha_post == pytest.approx(60.5)
    assert beta_post == pytest.approx(40.5)


def test_cabiqae_recovers_amplitude_ideal_regime():
    a_true = 0.36027
    executor = make_ideal_executor(a_true, seed=42)
    cabiqae = CABIQAE(c0=1.0, tau_c=1e12, b=0.5, rho_min=2.0)
    result = cabiqae.estimate(executor, epsilon=1e-2, alpha=0.1, n_batch=256, max_stages=60)
    assert abs(result.a_hat - a_true) < 0.05
    assert result.a_lower <= result.a_hat <= result.a_upper


def test_biqae_recovers_amplitude_ideal_regime():
    a_true = 0.5
    executor = make_ideal_executor(a_true, seed=7)
    biqae = BIQAE(rho_min=2.0)
    result = biqae.estimate(executor, epsilon=1e-2, alpha=0.1, n_batch=256, max_stages=60)
    assert abs(result.a_hat - a_true) < 0.05


def test_direct_circuit_sampling_returns_reasonable_estimate():
    a_true = 0.3
    executor = make_ideal_executor(a_true, seed=11)
    dcs = DirectCircuitSampling()
    result = dcs.estimate(executor, n_shots=5000)
    assert abs(result.a_hat - a_true) < 0.05


def test_bae_runs_and_returns_result():
    a_true = 0.4
    executor = make_ideal_executor(a_true, seed=13)
    bae = BAE(n_particles=100)
    result = bae.estimate(executor, epsilon=0.05, n_batch=256)
    assert 0.0 <= result.a_hat <= 1.0


def test_contrast_calibrator_fits_known_parameters():
    b = 0.5
    c0_true, tau_c_true = 1.0, 30.0
    a_true = 0.36
    theta_true = np.arcsin(np.sqrt(a_true))
    k_values = np.arange(0, 30)
    K = 2 * k_values + 1
    q_k = np.sin(K * theta_true) ** 2
    c_k = c0_true * np.exp(-K / tau_c_true)
    observed = b + c_k * (q_k - b)

    calibrator = ContrastCalibrator(b=b)
    c0_fit, tau_c_fit, r2 = calibrator.fit_contrast_model(k_values, observed, ideal_theta=theta_true)
    assert tau_c_fit == pytest.approx(tau_c_true, rel=0.1)
    assert r2 > 0.9


def test_contrast_calibrator_readout_mitigation():
    calibrator = ContrastCalibrator(b=0.5)
    mitigated = calibrator.readout_mitigate(np.array([0.6]), r0=0.01, r1=0.98)
    assert 0.0 <= mitigated[0] <= 1.0


def test_hardware_replay_model_executor():
    calibrator = ContrastCalibrator(b=0.5)
    hw_probs = {0: 0.4, 1: 0.3, 2: 0.2}
    executor = calibrator.build_hardware_replay_model(hw_probs, rng=np.random.default_rng(0))
    n_success, n_shots = executor(1, 1000)
    assert 0 <= n_success <= n_shots

    with pytest.raises(KeyError):
        executor(99, 100)
