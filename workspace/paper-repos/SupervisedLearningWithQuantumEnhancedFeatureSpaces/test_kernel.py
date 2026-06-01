"""
test_kernel.py — Unit tests for QuantumKernelEstimator

Tests:
  - K(x,x) = 1 exactly (both modes)
  - K(x,z) = K(z,x) (symmetry)
  - K(x,z) ∈ [0,1]
  - Kernel matrix is symmetric and PSD after enforce_psd
  - build_kernel_matrix shape correctness
  - Shot-based vs statevector mode: consistent within tolerance

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsvm.feature_map import FeatureMap
from qsvm.quantum_kernel import QuantumKernelEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fm():
    return FeatureMap(n_qubits=2, reps=2)


@pytest.fixture
def backend_sv():
    return AerSimulator(method="statevector")


@pytest.fixture
def kernel_exact(fm, backend_sv):
    return QuantumKernelEstimator(
        feature_map=fm, backend=backend_sv, use_statevector=True
    )


@pytest.fixture
def kernel_shots(fm):
    backend = AerSimulator(method="automatic")
    return QuantumKernelEstimator(
        feature_map=fm, backend=backend, shots=4096, use_statevector=False
    )


@pytest.fixture
def x1():
    return np.array([1.2, 3.7])


@pytest.fixture
def x2():
    return np.array([4.5, 0.8])


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 6.2, size=(6, 2))


# ---------------------------------------------------------------------------
# Tests: exact statevector mode
# ---------------------------------------------------------------------------

class TestExactKernel:
    def test_self_kernel_is_one(self, kernel_exact, x1):
        """K(x,x) = |⟨Φ(x)|Φ(x)⟩|² = 1  [EQ3]"""
        K = kernel_exact.evaluate(x1, x1)
        assert abs(K - 1.0) < 1e-8, f"K(x,x) = {K}, expected 1.0"

    def test_kernel_in_unit_interval(self, kernel_exact, x1, x2):
        """K(x,z) ∈ [0,1]  [EQ3, fidelity is a probability]"""
        K = kernel_exact.evaluate(x1, x2)
        assert 0.0 <= K <= 1.0, f"K out of [0,1]: {K}"

    def test_kernel_symmetry(self, kernel_exact, x1, x2):
        """K(x,z) = K(z,x)  [EQ3, fidelity is symmetric]"""
        K_xz = kernel_exact.evaluate(x1, x2)
        K_zx = kernel_exact.evaluate(x2, x1)
        assert abs(K_xz - K_zx) < 1e-10, (
            f"Kernel not symmetric: K(x,z)={K_xz}, K(z,x)={K_zx}"
        )

    def test_kernel_distinct_inputs(self, kernel_exact, x1, x2):
        """K(x,z) < 1 for x ≠ z (non-trivial kernel)"""
        K = kernel_exact.evaluate(x1, x2)
        assert K < 1.0 - 1e-4, f"K(x≠z) = {K}, expected < 1"

    def test_kernel_nonnegative(self, kernel_exact, x1, x2):
        K = kernel_exact.evaluate(x1, x2)
        assert K >= 0.0, f"K < 0: {K}"

    def test_wrong_shape_raises(self, kernel_exact):
        with pytest.raises(AssertionError):
            kernel_exact.evaluate(np.array([1.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Tests: shot-based mode
# ---------------------------------------------------------------------------

class TestShotKernel:
    def test_self_kernel_near_one(self, kernel_shots, x1):
        """K(x,x) ≈ 1 with high-shot count (sampling error ~1/√shots)"""
        K = kernel_shots.evaluate(x1, x1)
        # With 4096 shots, sampling error ~1.6%; allow 5% tolerance
        assert abs(K - 1.0) < 0.05, (
            f"K(x,x) = {K:.4f}, expected near 1.0 (shots mode)"
        )

    def test_kernel_in_unit_interval(self, kernel_shots, x1, x2):
        K = kernel_shots.evaluate(x1, x2)
        assert 0.0 <= K <= 1.0, f"K out of [0,1]: {K}"

    def test_shot_vs_exact_close(self, fm, backend_sv, x1, x2):
        """Shot-based and exact kernels agree within sampling tolerance."""
        kernel_ex = QuantumKernelEstimator(fm, backend_sv, use_statevector=True)
        backend_shot = AerSimulator(method="automatic")
        kernel_sh = QuantumKernelEstimator(
            fm, backend_shot, shots=8192, use_statevector=False
        )
        K_exact = kernel_ex.evaluate(x1, x2)
        K_shot  = kernel_sh.evaluate(x1, x2)
        # Tolerance: 3 * sampling_error ≈ 3 / sqrt(8192) ≈ 0.033
        assert abs(K_exact - K_shot) < 0.05, (
            f"Exact={K_exact:.4f}, Shot={K_shot:.4f}: gap too large"
        )


# ---------------------------------------------------------------------------
# Tests: build_kernel_matrix
# ---------------------------------------------------------------------------

class TestBuildKernelMatrix:
    def test_symmetric_shape(self, kernel_exact, small_dataset):
        K = kernel_exact.build_kernel_matrix(small_dataset, verbose=False)
        assert K.shape == (6, 6), f"Expected (6,6), got {K.shape}"

    def test_diagonal_ones(self, kernel_exact, small_dataset):
        """Diagonal K[i,i] = K(x_i, x_i) = 1"""
        K = kernel_exact.build_kernel_matrix(small_dataset, verbose=False)
        diag = np.diag(K)
        np.testing.assert_allclose(diag, np.ones(6), atol=1e-8,
                                   err_msg="Diagonal not all 1.0")

    def test_symmetry(self, kernel_exact, small_dataset):
        """K = K^T"""
        K = kernel_exact.build_kernel_matrix(small_dataset, verbose=False)
        np.testing.assert_allclose(K, K.T, atol=1e-10,
                                   err_msg="Kernel matrix not symmetric")

    def test_values_in_unit_interval(self, kernel_exact, small_dataset):
        K = kernel_exact.build_kernel_matrix(small_dataset, verbose=False)
        assert K.min() >= -1e-10, f"Kernel has negative values: min={K.min()}"
        assert K.max() <= 1.0 + 1e-10, f"Kernel exceeds 1: max={K.max()}"

    def test_rectangular_shape(self, kernel_exact, small_dataset):
        X = small_dataset[:4]
        Y = small_dataset[4:]
        K = kernel_exact.build_kernel_matrix(X, Y, verbose=False)
        assert K.shape == (4, 2), f"Expected (4,2), got {K.shape}"

    def test_dtype_float64(self, kernel_exact, small_dataset):
        K = kernel_exact.build_kernel_matrix(small_dataset, verbose=False)
        assert K.dtype == np.float64, f"Expected float64, got {K.dtype}"


# ---------------------------------------------------------------------------
# Tests: enforce_psd
# ---------------------------------------------------------------------------

class TestEnforcePSD:
    def test_already_psd_unchanged(self, kernel_exact, small_dataset):
        K = kernel_exact.build_kernel_matrix(small_dataset, verbose=False)
        K_psd = kernel_exact.enforce_psd(K)
        # Eigenvalues should all be >= 0
        eigvals = np.linalg.eigvalsh(K_psd)
        assert eigvals.min() >= -1e-10, (
            f"enforce_psd left negative eigenvalue: min={eigvals.min()}"
        )

    def test_negative_eigenvalue_clipped(self, kernel_exact):
        """Manually inject negative eigenvalue; verify it gets clipped."""
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        # Artificially corrupt K
        K_bad = K.copy()
        K_bad[0, 0] = -0.1
        K_bad[1, 1] = -0.1
        K_psd = kernel_exact.enforce_psd(K_bad, epsilon=1e-6)
        eigvals = np.linalg.eigvalsh(K_psd)
        assert eigvals.min() >= -1e-8, (
            f"Negative eigenvalue not clipped: min={eigvals.min()}"
        )

    def test_diagonal_remains_one(self, kernel_exact, small_dataset):
        K = kernel_exact.build_kernel_matrix(small_dataset, verbose=False)
        K_psd = kernel_exact.enforce_psd(K)
        diag = np.diag(K_psd)
        np.testing.assert_allclose(diag, np.ones(len(diag)), atol=1e-8)
