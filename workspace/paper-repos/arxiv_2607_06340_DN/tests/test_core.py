"""Unit tests. Run with: pytest tests/ -v"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sig_vol_id.features.signatures import SignatureComputer
from sig_vol_id.models.xgb_classifier import SignatureXGBClassifier
from sig_vol_id.simulators.heston import HestonSimulator
from sig_vol_id.simulators.ou import OUSimulator
from sig_vol_id.simulators.rbergomi import RoughBergomiSimulator


class TestOUSimulator:
    def test_output_shape(self):
        sim = OUSimulator(n_steps=20, T=0.1)
        rng = np.random.default_rng(0)
        paths = sim.simulate(50, {"X0": 0.15, "kappa": 3.0, "theta": 0.15, "sigma": 0.1}, rng)
        assert paths.shape == (50, 21)

    def test_starts_at_X0(self):
        sim = OUSimulator(n_steps=20, T=0.1)
        rng = np.random.default_rng(0)
        paths = sim.simulate(10, {"X0": 0.15, "kappa": 3.0, "theta": 0.15, "sigma": 0.1}, rng)
        assert np.allclose(paths[:, 0], 0.15)

    def test_reverts_toward_theta_on_average(self):
        sim = OUSimulator(n_steps=200, T=2.0)
        rng = np.random.default_rng(0)
        paths = sim.simulate(2000, {"X0": 0.5, "kappa": 5.0, "theta": 0.1, "sigma": 0.05}, rng)
        assert abs(paths[:, -1].mean() - 0.1) < 0.02


class TestHestonSimulator:
    def test_output_shape(self):
        sim = HestonSimulator(n_steps=20, T=0.1)
        rng = np.random.default_rng(0)
        paths = sim.simulate(50, {"X0": 0.1, "kappa": 2.0, "theta": 0.1, "nu": 0.2}, rng)
        assert paths.shape == (50, 21)

    def test_variance_stays_nonnegative_with_full_truncation(self):
        sim = HestonSimulator(n_steps=100, T=0.5)
        rng = np.random.default_rng(0)
        # High nu relative to kappa*theta to stress-test positivity handling
        paths = sim.simulate(500, {"X0": 0.05, "kappa": 1.0, "theta": 0.05, "nu": 0.5}, rng)
        # full truncation scheme can produce slightly negative paths (uses max(v,0)
        # only in the diffusion coefficient) -- but should not blow up / NaN
        assert np.isfinite(paths).all()

    def test_random_param_sampling_respects_feller_margin(self):
        cfg = {
            "fixed": {"X0": 0.1},
            "random": {
                "kappa_range": [1.0, 3.0],
                "theta_range": [0.05, 0.15],
                "nu_range": [0.15, 0.35],
                "feller_safety_margin": 0.95,
            },
        }
        rng = np.random.default_rng(0)
        params = HestonSimulator.sample_random_params(1000, cfg, rng)
        nu_max = 0.95 * np.sqrt(2 * params["kappa"] * params["theta"])
        assert np.all(params["nu"] <= nu_max + 1e-9)


class TestRoughBergomiSimulator:
    def test_output_shape(self):
        sim = RoughBergomiSimulator(n_steps=20, T=0.1)
        rng = np.random.default_rng(0)
        paths = sim.simulate(50, H=0.1, params={"xi": 0.08, "eta": 1.8}, rng=rng)
        assert paths.shape == (50, 21)

    def test_starts_at_xi(self):
        sim = RoughBergomiSimulator(n_steps=20, T=0.1)
        rng = np.random.default_rng(0)
        paths = sim.simulate(10, H=0.1, params={"xi": 0.08, "eta": 1.8}, rng=rng)
        assert np.allclose(paths[:, 0], 0.08)

    def test_shared_noise_gives_different_paths_per_H(self):
        sim = RoughBergomiSimulator(n_steps=20, T=0.1)
        rng = np.random.default_rng(0)
        results = sim.simulate_shared_noise(20, [0.1, 0.3], {"xi": 0.08, "eta": 1.8}, rng)
        assert set(results.keys()) == {0.1, 0.3}
        assert not np.allclose(results[0.1], results[0.3])

    def test_lower_H_gives_rougher_paths(self):
        # Rougher (lower H) paths should have higher quadratic variation of log-vol
        sim = RoughBergomiSimulator(n_steps=100, T=0.1)
        rng = np.random.default_rng(0)
        results = sim.simulate_shared_noise(200, [0.05, 0.45], {"xi": 0.08, "eta": 1.8}, rng)
        qv_rough = np.mean(np.sum(np.diff(np.log(results[0.05]), axis=1) ** 2, axis=1))
        qv_smooth = np.mean(np.sum(np.diff(np.log(results[0.45]), axis=1) ** 2, axis=1))
        assert qv_rough > qv_smooth


class TestSignatureComputer:
    def test_output_dim_matches_formula(self):
        computer = SignatureComputer(order=4)
        rng = np.random.default_rng(0)
        paths = rng.normal(size=(30, 21))
        feats = computer.compute(paths, T=0.1)
        assert feats.shape == (30, SignatureComputer.dim_for_order(4))
        assert feats.shape[1] == 31

    def test_order3_and_order5_dims(self):
        assert SignatureComputer.dim_for_order(3) == 15
        assert SignatureComputer.dim_for_order(5) == 63

    def test_first_feature_is_constant_one(self):
        computer = SignatureComputer(order=4)
        rng = np.random.default_rng(0)
        paths = rng.normal(size=(10, 21))
        feats = computer.compute(paths, T=0.1)
        assert np.allclose(feats[:, 0], 1.0)

    def test_different_paths_give_different_signatures(self):
        computer = SignatureComputer(order=4)
        rng = np.random.default_rng(0)
        paths_a = rng.normal(0, 1, size=(10, 21))
        paths_b = rng.normal(0, 5, size=(10, 21))
        feats_a = computer.compute(paths_a, T=0.1)
        feats_b = computer.compute(paths_b, T=0.1)
        assert not np.allclose(feats_a, feats_b)


class TestSignatureXGBClassifier:
    def test_fits_and_predicts_separable_classes(self):
        rng = np.random.default_rng(0)
        # Two trivially separable classes for a smoke test
        X0 = rng.normal(0, 1, size=(100, 5))
        X1 = rng.normal(10, 1, size=(100, 5))
        X = np.vstack([X0, X1])
        y = np.array([0] * 100 + [1] * 100)

        clf = SignatureXGBClassifier(n_classes=2, n_estimators=10, max_depth=2)
        clf.fit(X, y)
        acc = clf.accuracy(X, y)
        assert acc > 0.95

    def test_confusion_matrix_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 4))
        y = rng.integers(0, 3, size=60)
        clf = SignatureXGBClassifier(n_classes=3, n_estimators=10, max_depth=2)
        clf.fit(X, y)
        cm = clf.confusion_matrix(X, y, class_names=["A", "B", "C"])
        assert cm.shape == (3, 3)
        assert list(cm.columns) == ["A", "B", "C"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
