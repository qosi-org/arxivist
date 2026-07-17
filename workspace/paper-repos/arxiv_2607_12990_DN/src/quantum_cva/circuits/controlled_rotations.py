"""
Controlled Rotations Circuit Ansatz (CRCA) blocks: R_v, R_p, R_q.

Implements Section 3.2.3, Figures 4-5 of arXiv:2607.12990. Each CRCA block
approximates a deterministic map f_target: X_f -> [0,1] by training an
ancilla-conditional rotation:

    |0> |--> sqrt(1 - F_phi(x)) |0> + sqrt(F_phi(x)) |1>

via the mean-square loss (Section 3.2.3):

    L_f(phi) = (1/|X_f|) * sum_x (F_phi(x) - f_target(x))^2

Two topologies are implemented:
  - NativeTreeCRCA: 3-node path (t0 - ancilla - t1), used for R_p (discount
    factor) and R_q (default probability), which condition only on the time
    register (Figure 4).
  - SnakeCRCA: alternating outward/inward snake layers over the full
    time-market register plus ancilla, used for R_v (positive exposure),
    which must condition on the full (i,j) register because exposure is
    non-linear and non-separable across time and asset coordinates (Figure 5).

SIR reference: architecture.modules "R_v/R_p/R_q (CRCA)", mathematical_spec
"CRCA controlled-rotation MSE training loss".
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


class NativeTreeCRCA:
    """3-node-path CRCA for R_p (discount factor) and R_q (default probability).

    Topology (Figure 4): time qubits t0, t1 couple directly to a single
    ancilla via RX/RZ single-qubit rotations and RZZ entanglers, forming a
    3-node path graph that embeds into any heavy-hex 3-path with zero
    routing overhead (Table 11).

    Args:
        num_time_qubits: m, number of time-register qubits this block
            conditions on (2 for the paper's six-qubit instance).
    """

    def __init__(self, num_time_qubits: int) -> None:
        self.num_time_qubits = num_time_qubits
        self.num_qubits = num_time_qubits + 1  # + 1 ancilla

    def __repr__(self) -> str:  # noqa: D105
        return f"NativeTreeCRCA(num_time_qubits={self.num_time_qubits})"

    def build_circuit(self, num_layers: int) -> QuantumCircuit:
        """Build the native-tree CRCA circuit (Figure 4).

        Args:
            num_layers: L (paper uses L=1 for both R_p and R_q, Table 5).

        Returns:
            Parametrised QuantumCircuit; last qubit index is the ancilla.
        """
        qc = QuantumCircuit(self.num_qubits, name="CRCA_tree")
        ancilla = self.num_qubits - 1
        # Per-layer param budget: RX+RZ on ancilla (start), RZZ per time qubit
        # (down + up), RX+RZ on ancilla (end) -- matches Figure 4's structure
        # (RX,RZ -> ZZ -> ZZ -> RX,RZ for the 2-time-qubit case).
        n_params_per_layer = 2 + 2 * self.num_time_qubits + 2
        params = ParameterVector("phi", num_layers * n_params_per_layer)

        idx = 0
        for _layer in range(num_layers):
            qc.rx(params[idx], ancilla)
            idx += 1
            qc.rz(params[idx], ancilla)
            idx += 1
            for t in range(self.num_time_qubits):
                qc.rzz(params[idx], t, ancilla)
                idx += 1
            qc.rx(params[idx], ancilla)
            idx += 1
            qc.rz(params[idx], ancilla)
            idx += 1
        return qc

    def conditional_probability(
        self, circuit: QuantumCircuit, params: np.ndarray, time_index: int
    ) -> float:
        """Compute F_phi(i): probability the ancilla measures |1>, conditioned
        on the time register being in basis state |i>.

        Implemented by preparing the time register in |i> (via X gates on the
        relevant qubits prepended to the CRCA circuit) and reading the reduced
        ancilla density.

        Args:
            circuit: unbound circuit from `build_circuit`.
            params: flat parameter array.
            time_index: integer i in [0, 2**num_time_qubits).

        Returns:
            F_phi(i) in [0, 1].
        """
        prep = QuantumCircuit(self.num_qubits)
        bits = format(time_index, f"0{self.num_time_qubits}b")[::-1]
        for q, bit in enumerate(bits):
            if bit == "1":
                prep.x(q)
        full = prep.compose(circuit)
        bound = full.assign_parameters(params)
        state = Statevector.from_instruction(bound)
        ancilla = self.num_qubits - 1
        probs = state.probabilities_dict()
        p1 = sum(p for bitstr, p in probs.items() if bitstr[0] == "1")
        # Qiskit little-endian: bitstr[0] corresponds to the highest-indexed
        # qubit when read left-to-right; ancilla is qubit index self.num_qubits-1
        # which is the leftmost character in Qiskit's default string ordering.
        return float(p1) if ancilla == self.num_qubits - 1 else float(1 - p1)

    def mse_loss(
        self, circuit: QuantumCircuit, params: np.ndarray, target_fn: np.ndarray
    ) -> float:
        """MSE training loss L_f(phi) = mean((F_phi(i) - f_target(i))^2).

        Args:
            circuit: unbound circuit from `build_circuit`.
            params: flat parameter array.
            target_fn: array of target values f_target(i) for i in
                range(2**num_time_qubits), e.g. p~_i or q~_i.

        Returns:
            Scalar MSE loss.
        """
        n = 2**self.num_time_qubits
        preds = np.array(
            [self.conditional_probability(circuit, params, i) for i in range(n)]
        )
        return float(np.mean((preds - target_fn) ** 2))


class SnakeCRCA:
    """Outward/inward snake-layer CRCA for R_v (positive exposure).

    Topology (Figure 5): the exposure ancilla's effective receptive field is
    expanded across the full 7-qubit patch (6 register qubits + 1 ancilla)
    through alternating outward and inward "snake" sweeps of nearest-neighbour
    RX/RY/RZ + controlled interactions, respecting the heavy-hex maximum
    physical degree of 3 (Table 11) without requiring a literal degree-6 star.

    Layer repetition: the paper depicts one variational layer (L=1) explicitly
    and states the full circuit is "obtained by composing several repetitions
    of these layers, with independent parameters in each repetition." L=2 is
    used for the final R_v (Table 5): this implementation repeats the
    outward-then-inward snake pair with independent parameters per repetition
    -- see architecture_plan.json risk_assessment for the corresponding
    validation strategy (checking resulting gate counts against Table 5).

    Args:
        num_time_qubits: m
        num_asset_qubits: n
    """

    def __init__(self, num_time_qubits: int, num_asset_qubits: int) -> None:
        self.num_time_qubits = num_time_qubits
        self.num_asset_qubits = num_asset_qubits
        self.num_register_qubits = num_time_qubits + num_asset_qubits
        self.num_qubits = self.num_register_qubits + 1  # + exposure ancilla

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SnakeCRCA(m={self.num_time_qubits}, n={self.num_asset_qubits})"
        )

    def _snake_layer(
        self, qc: QuantumCircuit, params: ParameterVector, start_idx: int, outward: bool
    ) -> int:
        """Apply one outward or inward snake sweep; returns updated param index."""
        ancilla = self.num_qubits - 1
        order = list(range(self.num_register_qubits))
        if not outward:
            order = order[::-1]

        idx = start_idx
        # Local single-qubit rotations on the ancilla to seed the sweep
        qc.ry(params[idx], ancilla)
        idx += 1
        qc.rz(params[idx], ancilla)
        idx += 1

        prev = ancilla
        for q in order:
            qc.rzz(params[idx], prev, q)
            idx += 1
            qc.rx(params[idx], q)
            idx += 1
            qc.ry(params[idx], q)
            idx += 1
            prev = q
        # Route information back to the ancilla with a final entangler
        qc.rzz(params[idx], prev, ancilla)
        idx += 1
        return idx

    def build_circuit(self, num_layers: int) -> QuantumCircuit:
        """Build the snake-topology CRCA circuit for R_v (Figure 5).

        Args:
            num_layers: L (paper uses L=2 for the final six-qubit CVA
                instance, Table 5).

        Returns:
            Parametrised QuantumCircuit; last qubit index is the exposure
            ancilla a_v.
        """
        qc = QuantumCircuit(self.num_qubits, name="CRCA_snake_Rv")
        params_per_sweep = 2 + 3 * self.num_register_qubits + 1
        total_params = num_layers * 2 * params_per_sweep  # outward + inward per layer
        params = ParameterVector("phi_v", total_params)

        idx = 0
        for _layer in range(num_layers):
            idx = self._snake_layer(qc, params, idx, outward=True)
            qc.barrier()
            idx = self._snake_layer(qc, params, idx, outward=False)
            qc.barrier()
        return qc

    def conditional_probability(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        time_index: int,
        asset_index: tuple,
        asset_qubits_per_underlying: int,
    ) -> float:
        """Compute F_phi(i, j): ancilla |1> probability conditioned on the
        full register basis state (i, j).

        Args:
            circuit: unbound circuit from `build_circuit`.
            params: flat parameter array.
            time_index: integer i in [0, 2**num_time_qubits).
            asset_index: tuple (j_1, ..., j_d) of per-underlying bin indices.
            asset_qubits_per_underlying: n_k, assumed equal across underlyings
                (as in the paper's n1=n2=2 instance).

        Returns:
            F_phi(i, j) in [0, 1].
        """
        prep = QuantumCircuit(self.num_qubits)
        bits = format(time_index, f"0{self.num_time_qubits}b")[::-1]
        for q, bit in enumerate(bits):
            if bit == "1":
                prep.x(q)
        offset = self.num_time_qubits
        for jk in asset_index:
            jk_bits = format(jk, f"0{asset_qubits_per_underlying}b")[::-1]
            for q, bit in enumerate(jk_bits):
                if bit == "1":
                    prep.x(offset + q)
            offset += asset_qubits_per_underlying

        full = prep.compose(circuit)
        bound = full.assign_parameters(params)
        state = Statevector.from_instruction(bound)
        probs = state.probabilities_dict()
        p1 = sum(p for bitstr, p in probs.items() if bitstr[0] == "1")
        return float(p1)

    def mse_loss(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        target_tensor: np.ndarray,
        asset_qubits_per_underlying: int,
    ) -> float:
        """MSE loss L_v(phi) over the full finite grid of (i, j) points.

        Args:
            circuit: unbound circuit from `build_circuit`.
            params: flat parameter array.
            target_tensor: v~_{i,j}, shape [M, N_1, ..., N_d].
            asset_qubits_per_underlying: n_k (assumed equal for all underlyings).

        Returns:
            Scalar MSE loss.
        """
        shape = target_tensor.shape
        m_size = shape[0]
        asset_shape = shape[1:]
        sq_errors = []
        for i in range(m_size):
            for j_idx in np.ndindex(*asset_shape):
                pred = self.conditional_probability(
                    circuit, params, i, j_idx, asset_qubits_per_underlying
                )
                sq_errors.append((pred - target_tensor[(i,) + j_idx]) ** 2)
        return float(np.mean(sq_errors))
