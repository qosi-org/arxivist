"""
test_qke.py — Unit tests for QuantumKernelSVM and SyntheticQuantumDataset

Tests:
  - SyntheticQuantumDataset: shape, label balance, label validity, reproducibility
  - QuantumKernelSVM: fit runs, predict returns valid labels, score in [0,1]
  - Support vectors: non-empty, within training set bounds
  - Perfect separability: score=1.0 on training data (by construction)

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsvm.data import SyntheticQuantumDataset
from qsvm.feature_map import FeatureMap
from qsvm.kernel_svm import QuantumKernelSVM
from qsvm.quantum_kernel import QuantumKernelEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fm():
    return FeatureMap(n_qubits=2, reps=2)


@pytest.fixture
def backend():
    return AerSimulator(method="statevector")


@pytest.fixture
def kernel(fm, backend):
    return QuantumKernelEstimator(fm, backend, use_statevector=True)


@pytest.fixture
def small_dataset(fm):
    """5 points per label — fast for tests."""
    ds = SyntheticQuantumDataset(n_per_label=5, gap=0.3, seed=42)
    return ds.generate()


@pytest.fixture
def fitted_svm(kernel, small_dataset):
    X, y = small_dataset
    svm = QuantumKernelSVM(kernel_estimator=kernel, C=1.0)
    svm.fit(X, y, verbose=False)
    return svm, X, y


# ---------------------------------------------------------------------------
# Tests: SyntheticQuantumDataset
# ---------------------------------------------------------------------------

class TestSyntheticDataset:
    def test_output_shapes(self, small_dataset):
        X, y = small_dataset
        assert X.shape == (10, 2), f"Expected (10,2), got {X.shape}"
        assert y.shape == (10,),   f"Expected (10,), got {y.shape}"

    def test_label_balance(self, small_dataset):
        X, y = small_dataset
        assert (y == +1).sum() == 5, f"Expected 5 +1 labels, got {(y==+1).sum()}"
        assert (y == -1).sum() == 5, f"Expected 5 -1 labels, got {(y==-1).sum()}"

    def test_labels_valid(self, small_dataset):
        X, y = small_dataset
        assert set(y).issubset({+1, -1}), f"Invalid labels: {set(y)}"

    def test_domain_range(self, small_dataset):
        X, y = small_dataset
        assert X.min() > 0.0, "Data point below domain minimum"
        assert X.max() <= 2 * np.pi + 1e-6, "Data point above domain maximum"

    def test_reproducibility(self):
        ds1 = SyntheticQuantumDataset(n_per_label=5, gap=0.3, seed=42)
        ds2 = SyntheticQuantumDataset(n_per_label=5, gap=0.3, seed=42)
        X1, y1 = ds1.generate()
        X2, y2 = ds2.generate()
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        ds1 = SyntheticQuantumDataset(n_per_label=5, gap=0.3, seed=42)
        ds2 = SyntheticQuantumDataset(n_per_label=5, gap=0.3, seed=99)
        X1, _ = ds1.generate()
        X2, _ = ds2.generate()
        assert not np.allclose(X1, X2), "Different seeds produced identical data"

    def test_label_point_returns_valid(self, fm):
        ds = SyntheticQuantumDataset(n_per_label=5, gap=0.3, seed=42)
        ds.generate()
        x = np.array([1.0, 2.0])
        label = ds.label_point(x)
        assert label in (+1, -1, None), f"Invalid label: {label}"

    def test_gap_separation(self, fm):
        """All generated points must have |expectation| >= gap (by construction)."""
        ds = SyntheticQuantumDataset(n_per_label=10, gap=0.3, seed=42)
        X, y = ds.generate()
        backend = AerSimulator(method="statevector")
        for x_i, y_i in zip(X, y):
            sv = fm.get_statevector(x_i, backend)
            val = float(np.real(sv.conj() @ ds._VdagZ1Z2V @ sv))
            assert abs(val) >= 0.3 - 1e-6, (
                f"Point {x_i} has |val|={abs(val):.4f} < gap=0.3"
            )

    def test_split_returns_correct_shapes(self):
        ds = SyntheticQuantumDataset(n_per_label=5, gap=0.3, seed=42)
        X_tr, y_tr, X_te, y_te = ds.split(test_n_per_label=5)
        assert X_tr.shape == (10, 2)
        assert X_te.shape == (10, 2)
        assert y_tr.shape == (10,)
        assert y_te.shape == (10,)

    def test_invalid_gap_raises(self):
        with pytest.raises(ValueError, match="gap must be"):
            SyntheticQuantumDataset(gap=1.5)

    def test_invalid_n_per_label_raises(self):
        with pytest.raises(ValueError, match="n_per_label must be"):
            SyntheticQuantumDataset(n_per_label=0)


# ---------------------------------------------------------------------------
# Tests: QuantumKernelSVM
# ---------------------------------------------------------------------------

class TestQuantumKernelSVM:
    def test_fit_runs(self, fitted_svm):
        svm, X, y = fitted_svm
        assert svm._svm is not None

    def test_predict_valid_labels(self, fitted_svm):
        svm, X, y = fitted_svm
        y_pred = svm.predict(X)
        assert set(y_pred).issubset({+1, -1}), f"Invalid predicted labels: {set(y_pred)}"

    def test_predict_shape(self, fitted_svm):
        svm, X, y = fitted_svm
        y_pred = svm.predict(X)
        assert y_pred.shape == y.shape, (
            f"Predict shape mismatch: {y_pred.shape} vs {y.shape}"
        )

    def test_score_in_unit_interval(self, fitted_svm):
        svm, X, y = fitted_svm
        s = svm.score(X, y)
        assert 0.0 <= s <= 1.0, f"Score out of [0,1]: {s}"

    def test_training_score_perfect(self, fitted_svm):
        """
        The synthetic dataset is perfectly separable by construction.
        The QKE with exact kernel should achieve 100% on training data.
        """
        svm, X, y = fitted_svm
        s = svm.score(X, y)
        assert s == 1.0, (
            f"QKE training score={s:.2f}, expected 1.0 (perfectly separable data)"
        )

    def test_support_vectors_nonempty(self, fitted_svm):
        svm, X, y = fitted_svm
        svs, alphas, sv_labels = svm.get_support_vectors()
        assert len(svs) > 0, "No support vectors found"
        assert len(svs) == len(alphas) == len(sv_labels)

    def test_support_vectors_positive_alphas(self, fitted_svm):
        svm, X, y = fitted_svm
        _, alphas, _ = svm.get_support_vectors()
        assert (alphas >= 0).all(), f"Negative alpha found: {alphas.min()}"

    def test_decision_function_shape(self, fitted_svm):
        svm, X, y = fitted_svm
        vals = svm.decision_function_values(X)
        assert vals.shape == (len(X),), f"Decision function shape: {vals.shape}"

    def test_predict_before_fit_raises(self, kernel):
        svm = QuantumKernelSVM(kernel_estimator=kernel)
        with pytest.raises(RuntimeError, match="not fitted"):
            svm.predict(np.array([[1.0, 2.0]]))

    def test_get_bias_after_fit(self, fitted_svm):
        svm, _, _ = fitted_svm
        b = svm.get_bias()
        assert isinstance(b, float), f"Bias is not float: {type(b)}"
        assert np.isfinite(b), f"Bias is not finite: {b}"
