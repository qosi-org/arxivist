"""Unit tests for quantum_horizon.mining (mining competitiveness model)."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_horizon.mining import MiningCompetitivenessModel


def test_effective_hashrate_matches_calibration_point():
    m = MiningCompetitivenessModel()
    result = m.effective_hashrate(0.0667)  # the calibration gate speed itself, in GHz
    assert result == pytest.approx(0.0138, rel=1e-2)


def test_effective_hashrate_at_100ghz_matches_paper():
    """Paper states a single quantum machine at 100 GHz reaches ~21 TH/s."""
    m = MiningCompetitivenessModel()
    result = m.effective_hashrate(100)
    assert result == pytest.approx(21, rel=0.05)


def test_effective_hashrate_scales_linearly():
    m = MiningCompetitivenessModel()
    r1 = m.effective_hashrate(10)
    r2 = m.effective_hashrate(20)
    assert r2 == pytest.approx(2 * r1, rel=1e-6)


def test_effective_hashrate_vectorized():
    m = MiningCompetitivenessModel()
    speeds = np.array([1, 10, 100])
    results = m.effective_hashrate(speeds)
    assert results.shape == (3,)
    assert np.all(np.diff(results) > 0)


def test_parallel_hashrate_sqrt_scaling():
    """Grover parallelization wall: K machines give only sqrt(K) speedup."""
    m = MiningCompetitivenessModel()
    single = m.effective_hashrate(100)
    quad = m.parallel_hashrate(100, 4)
    assert quad == pytest.approx(2 * single, rel=1e-6)  # sqrt(4) = 2


def test_parallel_hashrate_far_below_linear_scaling():
    m = MiningCompetitivenessModel()
    single = m.effective_hashrate(100)
    parallel = m.parallel_hashrate(100, 10000)
    linear_equivalent = single * 10000
    assert parallel < linear_equivalent  # sqrt(K) << K for large K


def test_single_quantum_machine_far_below_one_asic():
    """Paper: a single quantum machine at 100 GHz is 'roughly one-tenth of
    a single modern ASIC' (~200 TH/s)."""
    m = MiningCompetitivenessModel()
    single = m.effective_hashrate(100)
    assert single < 200 / 5  # well below even a conservative ASIC estimate


def test_machines_for_target_hashrate_positive():
    m = MiningCompetitivenessModel()
    k = m.machines_for_target_hashrate(100, 1000)
    assert k > 0


def test_machines_for_51_percent_is_enormous():
    """Paper: reaching 51% of the 2026 network requires 'on the order of
    7x10^13 quantum machines at 100 GHz' -- our reconstructed formula
    (SIR confidence: ASSUMED, no explicit paper formula) gives a similar
    order of magnitude (within roughly one order of magnitude)."""
    m = MiningCompetitivenessModel()
    k = m.machines_for_51_percent(100, 860)
    assert 1e13 < k < 1e16
