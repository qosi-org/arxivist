"""Unit tests for ews_kalman.simulation (six tipping-point systems)."""

from __future__ import annotations

import numpy as np
import pytest

from ews_kalman.simulation import TippingSystemSimulator


def test_fold_bifurcation_shape_and_tipping_index():
    sim = TippingSystemSimulator()
    result = sim.fold_bifurcation(n_steps=250, tipping_t=200, seed=0)
    assert result["x"].shape == (250,)
    assert result["tipping_index"] == 200


def test_fold_bifurcation_r_crosses_zero_near_tipping():
    sim = TippingSystemSimulator()
    result = sim.fold_bifurcation(n_steps=250, tipping_t=200, seed=0)
    assert abs(result["r"][200]) < 0.05


def test_beta_step_change_shape_and_regime_values():
    sim = TippingSystemSimulator()
    result = sim.beta_step_change(n_steps=300, tipping_t=200, beta_before=0.8, beta_after=0.2, seed=0)
    assert result["x"].shape == (300,)
    assert result["y"].shape == (300,)
    assert np.all(result["beta_true"][:200] == 0.8)
    assert np.all(result["beta_true"][200:] == 0.2)


def test_beta_linear_decay_endpoints():
    sim = TippingSystemSimulator()
    result = sim.beta_linear_decay(n_steps=300, decay_end=250, beta_start=1.0, beta_end=0.2, seed=0)
    assert result["beta_true"][0] == pytest.approx(1.0, abs=1e-6)
    assert result["beta_true"][249] == pytest.approx(0.2, abs=1e-6)
    assert np.all(result["beta_true"][250:] == 0.2)


def test_logistic_map_values_in_unit_interval():
    sim = TippingSystemSimulator()
    result = sim.logistic_map(n_steps=400, seed=0)
    assert np.all(result["x"] >= 0.0) and np.all(result["x"] <= 1.0)


def test_logistic_map_r_schedule_increasing():
    sim = TippingSystemSimulator()
    result = sim.logistic_map(n_steps=400, r_start=2.5, r_end=4.0, seed=0)
    assert result["r"][0] == 2.5
    assert result["r"][-1] == 4.0
    assert np.all(np.diff(result["r"]) >= 0)


def test_stommel_amoc_shapes_and_psi_definition():
    sim = TippingSystemSimulator()
    result = sim.stommel_amoc(n_steps=300, seed=0)
    assert result["T"].shape == (300,)
    assert result["S"].shape == (300,)
    assert np.allclose(result["psi"], result["T"] - result["S"])


def test_critical_slowing_down_lambda_decreases_to_zero():
    sim = TippingSystemSimulator()
    result = sim.critical_slowing_down(n_steps=300, lambda_start=1.0, lambda_end=0.0, seed=0)
    assert result["lambda"][0] == 1.0
    assert result["lambda"][-1] == 0.0
    assert np.all(np.diff(result["lambda"]) <= 0)


def test_simulations_are_reproducible_with_same_seed():
    sim = TippingSystemSimulator()
    r1 = sim.fold_bifurcation(seed=42)
    r2 = sim.fold_bifurcation(seed=42)
    assert np.allclose(r1["x"], r2["x"])


def test_simulations_differ_with_different_seed():
    sim = TippingSystemSimulator()
    r1 = sim.critical_slowing_down(seed=1)
    r2 = sim.critical_slowing_down(seed=2)
    assert not np.allclose(r1["x"], r2["x"])
