"""Unit tests for quantum_cva.circuits (QCBM, CRCA, CVA oracle)."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cva.circuits import CVAOracle, NativeTreeCRCA, QCBMStatePreparation, SnakeCRCA


def test_qcbm_circuit_builds_and_qubit_count():
    qcbm = QCBMStatePreparation(num_time_qubits=2, num_asset_qubits=4)
    circuit = qcbm.build_circuit(num_layers=1)
    assert circuit.num_qubits == 6


def test_qcbm_born_distribution_sums_to_one():
    qcbm = QCBMStatePreparation(num_time_qubits=2, num_asset_qubits=4)
    circuit = qcbm.build_circuit(num_layers=1)
    rng = np.random.default_rng(0)
    params = rng.uniform(-np.pi, np.pi, circuit.num_parameters)
    dist = qcbm.born_distribution(circuit, params)
    assert dist.shape == (64,)
    assert np.isclose(dist.sum(), 1.0, atol=1e-8)


def test_qcbm_cross_entropy_nonnegative_at_uniform_target():
    qcbm = QCBMStatePreparation(num_time_qubits=1, num_asset_qubits=1)
    circuit = qcbm.build_circuit(num_layers=1)
    rng = np.random.default_rng(1)
    params = rng.uniform(-np.pi, np.pi, circuit.num_parameters)
    target = np.full(4, 0.25)
    loss = qcbm.cross_entropy_loss(circuit, params, target)
    assert loss > 0


def test_native_tree_crca_conditional_probability_bounds():
    r_p = NativeTreeCRCA(num_time_qubits=2)
    circuit = r_p.build_circuit(num_layers=1)
    rng = np.random.default_rng(2)
    params = rng.uniform(-np.pi, np.pi, circuit.num_parameters)
    for i in range(4):
        p = r_p.conditional_probability(circuit, params, i)
        assert 0.0 <= p <= 1.0


def test_native_tree_crca_mse_loss_zero_for_perfect_fit():
    r_p = NativeTreeCRCA(num_time_qubits=1)
    circuit = r_p.build_circuit(num_layers=1)
    rng = np.random.default_rng(3)
    params = rng.uniform(-np.pi, np.pi, circuit.num_parameters)
    target = np.array(
        [r_p.conditional_probability(circuit, params, i) for i in range(2)]
    )
    loss = r_p.mse_loss(circuit, params, target)
    assert loss == pytest.approx(0.0, abs=1e-10)


def test_snake_crca_qubit_count():
    r_v = SnakeCRCA(num_time_qubits=2, num_asset_qubits=4)
    circuit = r_v.build_circuit(num_layers=1)
    assert circuit.num_qubits == 7  # 6 register + 1 ancilla


def test_snake_crca_layer_scaling_matches_expected_param_growth():
    r_v = SnakeCRCA(num_time_qubits=2, num_asset_qubits=4)
    c1 = r_v.build_circuit(num_layers=1)
    c2 = r_v.build_circuit(num_layers=2)
    assert c2.num_parameters == 2 * c1.num_parameters


def test_cva_oracle_marked_amplitude_in_unit_interval():
    from qiskit import QuantumCircuit

    qcbm = QCBMStatePreparation(num_time_qubits=1, num_asset_qubits=1)
    r_v = SnakeCRCA(num_time_qubits=1, num_asset_qubits=1)
    r_p = NativeTreeCRCA(num_time_qubits=1)
    r_q = NativeTreeCRCA(num_time_qubits=1)

    rng = np.random.default_rng(4)
    g = qcbm.build_circuit(1).assign_parameters(
        rng.uniform(-np.pi, np.pi, qcbm.build_circuit(1).num_parameters)
    )
    rv = r_v.build_circuit(1).assign_parameters(
        rng.uniform(-np.pi, np.pi, r_v.build_circuit(1).num_parameters)
    )
    rp = r_p.build_circuit(1).assign_parameters(
        rng.uniform(-np.pi, np.pi, r_p.build_circuit(1).num_parameters)
    )
    rq = r_q.build_circuit(1).assign_parameters(
        rng.uniform(-np.pi, np.pi, r_q.build_circuit(1).num_parameters)
    )

    oracle = CVAOracle(num_register_qubits=2, num_ancillas=3)
    a_theta = oracle.assemble(g, rv, rp, rq)
    a_cva = oracle.marked_amplitude_statevector(a_theta)
    assert 0.0 <= a_cva <= 1.0


def test_grover_iterate_preserves_qubit_count():
    from qiskit import QuantumCircuit

    qcbm = QCBMStatePreparation(num_time_qubits=1, num_asset_qubits=1)
    r_v = SnakeCRCA(num_time_qubits=1, num_asset_qubits=1)
    r_p = NativeTreeCRCA(num_time_qubits=1)
    r_q = NativeTreeCRCA(num_time_qubits=1)

    rng = np.random.default_rng(5)
    g = qcbm.build_circuit(1).assign_parameters(rng.uniform(-np.pi, np.pi, qcbm.build_circuit(1).num_parameters))
    rv = r_v.build_circuit(1).assign_parameters(rng.uniform(-np.pi, np.pi, r_v.build_circuit(1).num_parameters))
    rp = r_p.build_circuit(1).assign_parameters(rng.uniform(-np.pi, np.pi, r_p.build_circuit(1).num_parameters))
    rq = r_q.build_circuit(1).assign_parameters(rng.uniform(-np.pi, np.pi, r_q.build_circuit(1).num_parameters))

    oracle = CVAOracle(num_register_qubits=2, num_ancillas=3)
    a_theta = oracle.assemble(g, rv, rp, rq)
    amplified = oracle.amplified_circuit(a_theta, k=2)
    assert amplified.num_qubits == a_theta.num_qubits
