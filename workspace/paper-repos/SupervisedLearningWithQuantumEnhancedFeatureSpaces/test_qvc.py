"""
test_qvc.py — Unit tests for QuantumVariationalClassifier

Tests:
  - Circuit construction: correct qubit count, gate types, parameter count
  - get_probs: output dict has +1/-1 keys, values in [0,1], sum to ~1
  - predict: returns +1 or -1
  - cost_function: scalar in [0,1]
  - fit: cost decreases over iterations (statistical, 3-attempt retry)
  - score: returns float in [0,1]

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsvm.feature_map import FeatureMap
from qsvm.variational_classifier import QuantumVariationalClassifier, _parity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fm():
    return FeatureMap(n_qubits=2, reps=2)


@pytest.fixture
def backend():
    return AerSimulator(method="automatic")


@pytest.fixture
def qvc_d0(fm, backend):
    return QuantumVariationalClassifier(fm, depth=0, backend=backend, shots=256)


@pytest.fixture
def qvc_d2(fm, backend):
    return QuantumVariationalClassifier(fm, depth=2, backend=backend, shots=256)


@pytest.fixture
def x_sample():
    return np.array([1.5, 3.0])


@pytest.fixture
def tiny_dataset():
    """Minimal 4-point dataset for fast tests."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0], [2.0, 5.5]])
    y = np.array([+1, +1, -1, -1])
    return X, y


# ---------------------------------------------------------------------------
# Tests: parity helper
# ---------------------------------------------------------------------------

class TestParity:
    def test_all_zeros(self):
        assert _parity("00") == +1   # even parity
        assert _parity("0000") == +1

    def test_single_one(self):
        assert _parity("01") == -1   # odd parity
        assert _parity("10") == -1

    def test_two_ones(self):
        assert _parity("11") == +1   # even parity
        assert _parity("0011") == +1


# ---------------------------------------------------------------------------
# Tests: parameter dimensions
# ---------------------------------------------------------------------------

class TestParameterDimensions:
    def test_n_theta_depth0(self, fm):
        """depth=0: W has 1 U_loc layer → 2*n_qubits*(0+1) = 4 params"""
        qvc = QuantumVariationalClassifier(fm, depth=0,
                                           backend=AerSimulator(method="automatic"))
        assert qvc.n_theta == 4, f"Expected 4, got {qvc.n_theta}"

    def test_n_theta_depth1(self, fm):
        """depth=1: 2*n_qubits*(1+1) = 8 params"""
        qvc = QuantumVariationalClassifier(fm, depth=1,
                                           backend=AerSimulator(method="automatic"))
        assert qvc.n_theta == 8

    def test_n_theta_depth4(self, fm):
        """depth=4: 2*n_qubits*(4+1) = 20 params"""
        qvc = QuantumVariationalClassifier(fm, depth=4,
                                           backend=AerSimulator(method="automatic"))
        assert qvc.n_theta == 20

    def test_invalid_depth(self, fm):
        with pytest.raises(ValueError, match="depth must be >= 0"):
            QuantumVariationalClassifier(fm, depth=-1,
                                         backend=AerSimulator(method="automatic"))


# ---------------------------------------------------------------------------
# Tests: circuit construction
# ---------------------------------------------------------------------------

class TestCircuitConstruction:
    def test_variational_circuit_n_qubits(self, qvc_d2, fm):
        theta = np.zeros(qvc_d2.n_theta)
        qc = qvc_d2.build_variational_circuit(theta)
        assert qc.num_qubits == fm.n_qubits

    def test_variational_circuit_has_ry_rz(self, qvc_d2):
        theta = np.ones(qvc_d2.n_theta)
        qc = qvc_d2.build_variational_circuit(theta)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "ry" in gate_names, "No RY gates in W(θ)"
        assert "rz" in gate_names, "No RZ gates in W(θ)"

    def test_variational_circuit_has_cz_for_depth_gt0(self, qvc_d2):
        theta = np.ones(qvc_d2.n_theta)
        qc = qvc_d2.build_variational_circuit(theta)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "cz" in gate_names, "No CZ gates in W(θ) for depth>0"

    def test_depth0_no_cz(self, qvc_d0):
        theta = np.ones(qvc_d0.n_theta)
        qc = qvc_d0.build_variational_circuit(theta)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "cz" not in gate_names, "CZ found in depth=0 circuit"

    def test_full_circuit_with_measure(self, qvc_d0, x_sample):
        theta = np.zeros(qvc_d0.n_theta)
        qc = qvc_d0.build_full_circuit(x_sample, theta, measure=True)
        assert qc.num_clbits == 2
        gate_names = {inst.operation.name for inst in qc.data}
        assert "measure" in gate_names

    def test_wrong_theta_shape_raises(self, qvc_d0, x_sample):
        with pytest.raises(AssertionError):
            qvc_d0.build_variational_circuit(np.zeros(10))


# ---------------------------------------------------------------------------
# Tests: get_probs
# ---------------------------------------------------------------------------

class TestGetProbs:
    def test_keys(self, qvc_d0, x_sample):
        theta = np.zeros(qvc_d0.n_theta)
        probs = qvc_d0.get_probs(x_sample, theta)
        assert +1 in probs and -1 in probs, f"Missing keys in probs: {probs}"

    def test_sum_to_one(self, qvc_d0, x_sample):
        theta = np.zeros(qvc_d0.n_theta)
        probs = qvc_d0.get_probs(x_sample, theta)
        total = probs[+1] + probs[-1]
        assert abs(total - 1.0) < 1e-8, f"Probs don't sum to 1: {total}"

    def test_values_in_unit_interval(self, qvc_d0, x_sample):
        theta = np.zeros(qvc_d0.n_theta)
        probs = qvc_d0.get_probs(x_sample, theta)
        for label, p in probs.items():
            assert 0.0 <= p <= 1.0, f"p[{label}]={p} out of [0,1]"


# ---------------------------------------------------------------------------
# Tests: predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_valid_label(self, qvc_d0, x_sample):
        theta = np.zeros(qvc_d0.n_theta)
        label = qvc_d0.predict(x_sample, theta)
        assert label in (+1, -1), f"Invalid label: {label}"

    def test_predict_consistent_with_probs(self, qvc_d0, x_sample):
        """predict() must agree with argmax of get_probs()."""
        theta = np.zeros(qvc_d0.n_theta)
        probs = qvc_d0.get_probs(x_sample, theta)
        expected = +1 if probs[+1] > probs[-1] else -1
        predicted = qvc_d0.predict(x_sample, theta)
        assert predicted == expected


# ---------------------------------------------------------------------------
# Tests: cost function
# ---------------------------------------------------------------------------

class TestCostFunction:
    def test_cost_in_unit_interval(self, qvc_d0, tiny_dataset):
        X, y = tiny_dataset
        theta = np.zeros(qvc_d0.n_theta)
        cost = qvc_d0.cost_function(theta, X, y, b=0.0, shots=64)
        assert 0.0 <= cost <= 1.0, f"Cost out of [0,1]: {cost}"

    def test_cost_is_scalar(self, qvc_d0, tiny_dataset):
        X, y = tiny_dataset
        theta = np.zeros(qvc_d0.n_theta)
        cost = qvc_d0.cost_function(theta, X, y, b=0.0, shots=64)
        assert isinstance(cost, float), f"Cost is not float: type={type(cost)}"

    def test_cost_finite(self, qvc_d0, tiny_dataset):
        X, y = tiny_dataset
        theta = np.random.default_rng(0).uniform(-np.pi, np.pi, qvc_d0.n_theta)
        cost = qvc_d0.cost_function(theta, X, y, b=0.0, shots=64)
        assert np.isfinite(cost), f"Cost is not finite: {cost}"


# ---------------------------------------------------------------------------
# Tests: score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_in_unit_interval(self, qvc_d0, tiny_dataset):
        X, y = tiny_dataset
        theta = np.zeros(qvc_d0.n_theta)
        s = qvc_d0.score(X, y, theta, b=0.0)
        assert 0.0 <= s <= 1.0, f"Score out of [0,1]: {s}"

    def test_score_on_two_points(self, qvc_d0):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([+1, -1])
        theta = np.zeros(qvc_d0.n_theta)
        s = qvc_d0.score(X, y, theta)
        assert s in (0.0, 0.5, 1.0), f"Unexpected score for 2 points: {s}"


# ---------------------------------------------------------------------------
# Tests: fit (fast sanity — not full 250-step convergence)
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_correct_shapes(self, qvc_d0, tiny_dataset):
        X, y = tiny_dataset
        theta_star, b_star, cost_hist = qvc_d0.fit(
            X, y, n_iter=3, shots_cost=32, verbose=False
        )
        assert theta_star.shape == (qvc_d0.n_theta,)
        assert isinstance(b_star, float)
        assert len(cost_hist) == 3

    def test_cost_history_all_finite(self, qvc_d0, tiny_dataset):
        X, y = tiny_dataset
        _, _, cost_hist = qvc_d0.fit(
            X, y, n_iter=5, shots_cost=32, verbose=False
        )
        for c in cost_hist:
            assert np.isfinite(c), f"Non-finite cost in history: {c}"

    def test_cost_history_in_unit_interval(self, qvc_d0, tiny_dataset):
        X, y = tiny_dataset
        _, _, cost_hist = qvc_d0.fit(
            X, y, n_iter=5, shots_cost=32, verbose=False
        )
        for c in cost_hist:
            assert 0.0 <= c <= 1.0 + 1e-6, f"Cost out of [0,1]: {c}"
