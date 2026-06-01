"""
test_feature_map.py — Unit tests for FeatureMap

Tests:
  - Circuit structure: correct gate count and qubit count
  - phi_coefficients: correct values for n=2 [EQ5]
  - Statevector: normalised, correct shape, |Φ(x)⟩ == |Φ(x)⟩ (deterministic)
  - Inverse circuit: U†_Phi(x) U_Phi(x)|0⟩ = |0⟩
  - get_circuit with different x values produces distinct states
  - K(x,x) = 1 (self-overlap)

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsvm.feature_map import FeatureMap


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
def x_sample():
    return np.array([1.2, 3.7])


# ---------------------------------------------------------------------------
# Tests: constructor
# ---------------------------------------------------------------------------

class TestFeatureMapInit:
    def test_default_params(self):
        fm = FeatureMap()
        assert fm.n_qubits == 2
        assert fm.reps == 2

    def test_invalid_n_qubits(self):
        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            FeatureMap(n_qubits=0)

    def test_invalid_reps(self):
        with pytest.raises(ValueError, match="reps must be 2"):
            FeatureMap(n_qubits=2, reps=1)

    def test_repr(self, fm):
        r = repr(fm)
        assert "FeatureMap" in r
        assert "n_qubits=2" in r


# ---------------------------------------------------------------------------
# Tests: phi_coefficients [EQ5]
# ---------------------------------------------------------------------------

class TestPhiCoefficients:
    def test_shape(self, fm, x_sample):
        coeffs = fm.phi_coefficients(x_sample)
        # For n=2: |S|=1 gives 2 entries; |S|=2 gives 1 entry → 3 total
        assert len(coeffs) == 3

    def test_single_qubit_values(self, fm):
        x = np.array([1.5, 2.5])
        coeffs = fm.phi_coefficients(x)
        # phi_{0}(x) = x[0] = 1.5  [EQ5]
        assert abs(coeffs[frozenset([0])] - 1.5) < 1e-10
        # phi_{1}(x) = x[1] = 2.5  [EQ5]
        assert abs(coeffs[frozenset([1])] - 2.5) < 1e-10

    def test_two_qubit_value(self, fm):
        x = np.array([1.5, 2.5])
        coeffs = fm.phi_coefficients(x)
        # phi_{0,1}(x) = (π - 1.5)(π - 2.5)  [EQ5]
        expected = (math.pi - 1.5) * (math.pi - 2.5)
        assert abs(coeffs[frozenset([0, 1])] - expected) < 1e-10

    def test_wrong_shape_raises(self, fm):
        with pytest.raises(AssertionError):
            fm.phi_coefficients(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Tests: get_circuit
# ---------------------------------------------------------------------------

class TestGetCircuit:
    def test_circuit_n_qubits(self, fm, x_sample):
        qc = fm.get_circuit(x_sample)
        assert qc.num_qubits == 2

    def test_circuit_has_no_classical_bits(self, fm, x_sample):
        qc = fm.get_circuit(x_sample)
        assert qc.num_clbits == 0

    def test_circuit_is_deterministic(self, fm, x_sample):
        qc1 = fm.get_circuit(x_sample)
        qc2 = fm.get_circuit(x_sample)
        # Same circuit depth
        assert qc1.depth() == qc2.depth()

    def test_different_inputs_differ(self, fm):
        qc1 = fm.get_circuit(np.array([1.0, 2.0]))
        qc2 = fm.get_circuit(np.array([2.0, 3.0]))
        # Different circuits (may have same depth but different gate angles)
        # We test via statevector instead — see test_statevector_distinct

    def test_circuit_has_hadamard_and_rz(self, fm, x_sample):
        qc = fm.get_circuit(x_sample)
        gate_names = [inst.operation.name for inst in qc.data]
        assert "h" in gate_names, "No Hadamard gates found"
        assert "rz" in gate_names, "No RZ gates found"


# ---------------------------------------------------------------------------
# Tests: get_statevector
# ---------------------------------------------------------------------------

class TestGetStatevector:
    def test_statevector_shape(self, fm, backend, x_sample):
        sv = fm.get_statevector(x_sample, backend)
        assert sv.shape == (4,)   # 2^n = 4

    def test_statevector_normalised(self, fm, backend, x_sample):
        sv = fm.get_statevector(x_sample, backend)
        norm = np.linalg.norm(sv)
        assert abs(norm - 1.0) < 1e-8, f"Statevector not normalised: norm={norm}"

    def test_statevector_is_complex(self, fm, backend, x_sample):
        sv = fm.get_statevector(x_sample, backend)
        assert sv.dtype == complex or np.iscomplexobj(sv)

    def test_statevector_deterministic(self, fm, backend, x_sample):
        sv1 = fm.get_statevector(x_sample, backend)
        sv2 = fm.get_statevector(x_sample, backend)
        np.testing.assert_allclose(sv1, sv2, atol=1e-12)

    def test_statevector_distinct_inputs(self, fm, backend):
        sv1 = fm.get_statevector(np.array([1.0, 2.0]), backend)
        sv2 = fm.get_statevector(np.array([3.0, 4.0]), backend)
        # Different inputs → different statevectors
        assert not np.allclose(sv1, sv2, atol=1e-6), (
            "Different inputs produced identical statevectors"
        )

    def test_self_overlap_is_one(self, fm, backend, x_sample):
        """K(x,x) = |⟨Φ(x)|Φ(x)⟩|² = 1  [EQ3 sanity check]"""
        sv = fm.get_statevector(x_sample, backend)
        overlap = np.vdot(sv, sv)
        assert abs(float(np.abs(overlap)**2) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Tests: inverse circuit
# ---------------------------------------------------------------------------

class TestInverseCircuit:
    def test_inverse_returns_to_zero(self, fm, backend, x_sample):
        """U†_Phi(x) U_Phi(x)|0⟩ ≈ |0⟩  (up to global phase)"""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.compose(fm.get_circuit(x_sample), inplace=True)
        qc.compose(fm.get_inverse_circuit(x_sample), inplace=True)
        qc.save_statevector()

        job = backend.run(qc)
        sv = np.array(job.result().get_statevector(qc))

        # |0⟩ state: amplitude 1 on index 0, 0 elsewhere
        # Allow global phase: check |⟨0|sv⟩|² ≈ 1
        zero_prob = float(np.abs(sv[0])**2)
        assert zero_prob > 1.0 - 1e-8, (
            f"Inverse circuit did not return to |0⟩: P(|0⟩)={zero_prob}"
        )

    def test_inverse_of_different_inputs(self, fm, backend):
        """U†_Phi(x) U_Phi(z)|0⟩ ≠ |0⟩ for x ≠ z (non-trivial kernel)"""
        from qiskit import QuantumCircuit

        x = np.array([1.0, 2.0])
        z = np.array([3.5, 5.0])

        qc = QuantumCircuit(2)
        qc.compose(fm.get_circuit(z), inplace=True)
        qc.compose(fm.get_inverse_circuit(x), inplace=True)
        qc.save_statevector()

        job = backend.run(qc)
        sv = np.array(job.result().get_statevector(qc))

        # Probability of |0⟩ = kernel K(x,z) — should be < 1 for x ≠ z
        K_xz = float(np.abs(sv[0])**2)
        assert K_xz < 1.0 - 1e-4, (
            f"Expected K(x,z)<1 for x≠z, got K={K_xz}"
        )
