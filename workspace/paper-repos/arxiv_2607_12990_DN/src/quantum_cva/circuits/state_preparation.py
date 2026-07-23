"""
QCBM (Quantum Circuit Born Machine) state-preparation ansatz G_theta.

Implements Section 3.2.3 and Figure 3 of arXiv:2607.12990. The trained
circuit approximates the state-preparation unitary G_P defined in Eq. 33:

    G_P |0>^{(m+n)} = sum_{i,j} sqrt(P_{i,j}) |i>_T |j_1>_{S_1} ... |j_d>_{S_d}

by minimising the regularised cross-entropy loss (Section 3.2.3):

    L_QCBM(theta) = - sum_x P_target(x) log(P_theta(x) + eps_num)

SIR reference: architecture.modules[0] "G_P / G_theta (QCBM state-preparation
unitary)", mathematical_spec "QCBM regularised cross-entropy training loss".
"""

from __future__ import annotations


import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


class QCBMStatePreparation:
    """Hardware-compatible QCBM ansatz for the joint time-market distribution.

    Architecture (Figure 3): each of `num_layers` blocks applies local RX, RZ
    rotations on every qubit, followed by a sparse RZZ entangler layer whose
    connectivity is fixed to a heavy-hex-embeddable graph:
        EQCBM = {(t0,t1), (t0,s0), (t0,s1), (t1,s2), (s0,s3)}   (Eq. 58)
    This graph couples the time register directly to both asset sub-registers
    for L=1; layers L=2..num_layers repeat the same pattern with independent
    parameters (paper: "composing several repetitions of these layers, with
    independent parameters in each repetition").

    Args:
        num_time_qubits: m, the number of time-register qubits.
        num_asset_qubits: n, total asset-register qubits (sum over underlyings).
    """

    def __init__(self, num_time_qubits: int, num_asset_qubits: int) -> None:
        self.num_time_qubits = num_time_qubits
        self.num_asset_qubits = num_asset_qubits
        self.num_qubits = num_time_qubits + num_asset_qubits

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"QCBMStatePreparation(m={self.num_time_qubits}, "
            f"n={self.num_asset_qubits})"
        )

    def _entangler_graph(self) -> list:
        """Sparse RZZ connectivity graph (Eq. 58), generalised to n qubits.

        For the paper's specific six-qubit instance (m=2, n=4) this reduces
        exactly to Eq. 58: {(t0,t1),(t0,s0),(t0,s1),(t1,s2),(s0,s3)}. For
        other qubit counts we generalise the same *pattern* (time qubits
        each couple to two asset qubits, remaining asset qubits couple in a
        chain) to preserve the zero-SWAP heavy-hex-embedding property
        described in Appendix A.2.
        """
        t = list(range(self.num_time_qubits))
        s = list(range(self.num_time_qubits, self.num_qubits))
        edges = []
        if self.num_time_qubits >= 2 and len(s) >= 4:
            # Exact Eq. 58 six-qubit case (m=2, n=4)
            edges = [(t[0], t[1]), (t[0], s[0]), (t[0], s[1]), (t[1], s[2]), (s[0], s[3])]
        else:
            # Generalisation: chain time qubits, then attach asset qubits round-robin
            for i in range(len(t) - 1):
                edges.append((t[i], t[i + 1]))
            for idx, sq in enumerate(s):
                anchor = t[idx % len(t)] if t else s[0]
                if anchor != sq:
                    edges.append((anchor, sq))
        return edges

    def build_circuit(self, num_layers: int) -> QuantumCircuit:
        """Build the parametrised QCBM circuit G_theta (Figure 3).

        Args:
            num_layers: L, the number of variational layers (paper uses L=6
                for the final six-qubit CVA instance, Table 5).

        Returns:
            A Qiskit QuantumCircuit with unbound Parameters. Bind via
            `circuit.assign_parameters(params_dict)` before simulation.
        """
        qc = QuantumCircuit(self.num_qubits, name="G_theta")
        edges = self._entangler_graph()
        n_params_per_layer = 2 * self.num_qubits + len(edges)
        params = ParameterVector("theta", num_layers * n_params_per_layer)

        idx = 0
        for _layer in range(num_layers):
            for q in range(self.num_qubits):
                qc.rx(params[idx], q)
                idx += 1
                qc.rz(params[idx], q)
                idx += 1
            for (a, b) in edges:
                qc.rzz(params[idx], a, b)
                idx += 1
            qc.barrier()
        return qc

    def born_distribution(
        self, circuit: QuantumCircuit, params: np.ndarray
    ) -> np.ndarray:
        """Compute the Born distribution P_theta(x) = |<x|G_theta|0>|^2.

        Args:
            circuit: unbound circuit from `build_circuit`.
            params: flat array of parameter values, length == circuit.num_parameters.

        Returns:
            Probability array of shape [2**num_qubits], statevector-exact
            (noiseless). For finite-shot/noisy evaluation, use a sampler
            backend via `hardware.backend_manager.BackendManager` instead.
        """
        bound = circuit.assign_parameters(params)
        state = Statevector.from_instruction(bound)
        return np.abs(state.data) ** 2

    def cross_entropy_loss(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        target_dist: np.ndarray,
        eps_num: float = 1e-8,
    ) -> float:
        """Regularised cross-entropy loss L_QCBM(theta) (Section 3.2.3).

        L_QCBM(theta) = - sum_x P_target(x) log(P_theta(x) + eps_num)

        Args:
            circuit: unbound circuit from `build_circuit`.
            params: flat parameter array.
            target_dist: P_target, shape [2**num_qubits], must sum to 1.
            eps_num: numerical floor to avoid log(0). ASSUMED value in
                config.yaml (paper states eps_num > 0 "small" but does not
                give the exact number -- SIR confidence < 0.7 for this
                constant).

        Returns:
            Scalar loss value.
        """
        p_theta = self.born_distribution(circuit, params)
        return -float(np.sum(target_dist * np.log(p_theta + eps_num)))

    def kl_divergence(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        target_dist: np.ndarray,
        eps_num: float = 1e-8,
    ) -> float:
        """Numerically regularised KL diagnostic KL_eps(P_target || P_theta).

        This is a *diagnostic metric* used to assess trained-circuit quality
        (Section 3.2.3), not the training objective itself (that is
        `cross_entropy_loss`).

        KL_eps(P_target||P_theta) = sum_{x: P_target(x)>0}
            P_target(x) * log(P_target(x) / (P_theta(x) + eps_num))
        """
        p_theta = self.born_distribution(circuit, params)
        mask = target_dist > 0
        return float(
            np.sum(
                target_dist[mask]
                * np.log(target_dist[mask] / (p_theta[mask] + eps_num))
            )
        )
