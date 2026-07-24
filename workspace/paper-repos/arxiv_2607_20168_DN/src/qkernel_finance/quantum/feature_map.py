"""
Quantum feature-map encoding circuit (Sec 4.2, Eq. 1).

    |psi(x)> = prod_{r=1}^{R} [ prod_q e^{-i x~_q x~_{q+1} Z_q Z_{q+1}}
                                 prod_q e^{-i x~_q Z_q H_q} ] |0>^{(x)n}

with n=8 qubits, R=2 repetitions, and ring entanglement (indices mod n).

*** HIGHEST-RISK ASSUMPTION IN THIS REPOSITORY ***
The paper never defines H_q. The literal notation -- Z_q and H_q co-occurring
in one exponent -- is not standard circuit shorthand. We implement the most
plausible standard reading (SIR ambiguities[0], confidence 0.6): H_q denotes
Hadamard conjugation of the single-qubit RZ rotation, i.e.

    e^{-i x~_q Z_q H_q}  :=  H_q . RZ(2*x~_q) . H_q  =  RX(2*x~_q)  (up to global phase)

which is the standard single-qubit term in IQP-style embeddings (Havlicek et
al. 2019). This is swappable in `_single_qubit_layer()` below -- if you have
reason to believe a different gate was intended (see `config.quantum.
hq_gate_interpretation` and architecture_plan.json's risk_assessment), change
it there; nothing else in this file needs to change.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


class QuantumFeatureMap:
    """The paper's n=8-qubit, R=2, ring-entangled IQP-style encoding circuit.

    Args:
        num_qubits: number of qubits (8, per Sec 4.2).
        repetitions: number of circuit repetitions R (2, per Sec 4.2).
        hq_gate_interpretation: which reading of the undefined H_q term to use.
            "hadamard_conjugated_rz" (default, our primary SIR reading) applies
            RX(2*x~_q) via H-RZ-H conjugation. See module docstring.
    """

    def __init__(self, num_qubits: int = 8, repetitions: int = 2, hq_gate_interpretation: str = "hadamard_conjugated_rz") -> None:
        self.num_qubits = num_qubits
        self.repetitions = repetitions
        self.hq_gate_interpretation = hq_gate_interpretation
        self.device = qml.device("default.qubit", wires=num_qubits)
        self._state_qnode = qml.QNode(self._circuit_statevector, self.device)

    def _single_qubit_layer(self, x_tilde: np.ndarray) -> None:
        """The prod_q e^{-i x~_q Z_q H_q} term -- see module docstring for the H_q ambiguity."""
        if self.hq_gate_interpretation == "hadamard_conjugated_rz":
            for q in range(self.num_qubits):
                qml.Hadamard(wires=q)
                qml.RZ(2.0 * x_tilde[q], wires=q)
                qml.Hadamard(wires=q)
        elif self.hq_gate_interpretation == "separate_hadamard_layer":
            # Alternative reading: a plain Hadamard layer, then an independent RZ layer.
            for q in range(self.num_qubits):
                qml.Hadamard(wires=q)
            for q in range(self.num_qubits):
                qml.RZ(2.0 * x_tilde[q], wires=q)
        else:
            raise ValueError(f"Unknown hq_gate_interpretation: {self.hq_gate_interpretation}")

    def _entangling_layer(self, x_tilde: np.ndarray) -> None:
        """The prod_q e^{-i x~_q x~_{q+1} Z_q Z_{q+1}} term, ring topology (indices mod n)."""
        n = self.num_qubits
        for q in range(n):
            q_next = (q + 1) % n
            angle = 2.0 * x_tilde[q] * x_tilde[q_next]
            qml.IsingZZ(angle, wires=[q, q_next])

    def _circuit_statevector(self, x_tilde: np.ndarray):
        for _ in range(self.repetitions):
            self._entangling_layer(x_tilde)
            self._single_qubit_layer(x_tilde)
        return qml.state()

    def state(self, x_tilde: np.ndarray) -> np.ndarray:
        """
        Args:
            x_tilde: [num_qubits] bandwidth-scaled input.

        Returns:
            [2**num_qubits] complex statevector |psi(x)>.
        """
        assert x_tilde.shape == (self.num_qubits,), f"Expected [{self.num_qubits}], got {x_tilde.shape}"
        return np.asarray(self._state_qnode(x_tilde))

    def states_batch(self, X_tilde: np.ndarray) -> np.ndarray:
        """
        Args:
            X_tilde: [N, num_qubits] batch of bandwidth-scaled inputs.

        Returns:
            [N, 2**num_qubits] complex statevectors.
        """
        return np.stack([self.state(row) for row in X_tilde], axis=0)

    def bloch_vectors(self, x_tilde: np.ndarray) -> np.ndarray:
        """Per-qubit Bloch vector (<X_q>, <Y_q>, <Z_q>) for the projected quantum kernel.

        Args:
            x_tilde: [num_qubits] bandwidth-scaled input.

        Returns:
            [3 * num_qubits] flattened Bloch feature vector phi(x) (24-dim for n=8).
        """
        psi = self.state(x_tilde)
        n = self.num_qubits
        feats = np.zeros(3 * n, dtype=np.float64)
        for q in range(n):
            feats[3 * q + 0] = self._expectation(psi, q, "X")
            feats[3 * q + 1] = self._expectation(psi, q, "Y")
            feats[3 * q + 2] = self._expectation(psi, q, "Z")
        return feats

    def _expectation(self, psi: np.ndarray, qubit: int, pauli: str) -> float:
        """Computes <psi| P_qubit |psi> for P in {X, Y, Z} via reduced density matrix."""
        n = self.num_qubits
        dim = 2**n
        psi = psi.reshape([2] * n)
        # Move target qubit to the front, trace out the rest to get the 2x2 reduced density matrix.
        axes = [qubit] + [q for q in range(n) if q != qubit]
        psi_perm = np.transpose(psi, axes).reshape(2, -1)
        rho = psi_perm @ psi_perm.conj().T  # [2,2] reduced density matrix
        paulis = {
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }
        return float(np.real(np.trace(rho @ paulis[pauli])))

    def __repr__(self) -> str:  # noqa: D105
        return f"QuantumFeatureMap(n={self.num_qubits}, R={self.repetitions}, hq={self.hq_gate_interpretation})"
