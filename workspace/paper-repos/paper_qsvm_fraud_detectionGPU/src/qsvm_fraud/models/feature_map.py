"""
models/feature_map.py — Quantum feature map for QSVM.

Implements the parameterized quantum circuit U_phi(x) that encodes classical
data into quantum states. The paper (Section III-A) uses Qiskit's ZZFeatureMap,
which applies Hadamard gates followed by ZZ-entangling layers.

Paper reference: Section III-A, Eq. 3
    |phi(x)> = U_phi(x)|0>^(⊗n)

Key assumptions (from SIR):
  - reps=2          (Qiskit default; confidence 0.60) — ASSUMED
  - entanglement='full' (Qiskit default; confidence 0.60) — ASSUMED
  - n_qubits equals n_features (stated explicitly in paper)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not installed. QSVMFeatureMap will raise on use.")


class QSVMFeatureMap:
    """
    Wrapper around Qiskit's ZZFeatureMap for use in QSVM.

    The ZZFeatureMap encodes classical data x ∈ ℝ^d into a quantum state
    |phi(x)> = U_phi(x)|0>^(⊗n) in a 2^n-dimensional Hilbert space.

    The circuit consists of:
      1. Layer of Hadamard gates
      2. Data-encoding single-qubit rotations: exp(i * x_j * Z_j)
      3. Entangling ZZ-rotations: exp(i * x_i * x_j * Z_i ⊗ Z_j)
      4. Repeated `reps` times

    Paper: Section III-A-2, Eq. 3. Explicitly names ZZFeatureMap (Section IV-B).

    Args:
        n_qubits:     Number of qubits == number of features (confidence: 0.97).
        reps:         Number of circuit repetitions. ASSUMED=2 (Qiskit default,
                      confidence: 0.60). TODO: verify against paper source.
        entanglement: Entanglement pattern. ASSUMED='full' (Qiskit default,
                      confidence: 0.60). Options: 'full','linear','circular','sca'.
    """

    def __init__(
        self,
        n_qubits: int = 10,
        reps: int = 2,              # ASSUMED: Qiskit default (confidence: 0.60)
        entanglement: str = "full", # ASSUMED: Qiskit default (confidence: 0.60)
    ) -> None:
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "qiskit and qiskit-machine-learning are required. "
                "Run: pip install qiskit qiskit-machine-learning qiskit-aer"
            )
        if n_qubits not in (4, 8, 10):
            raise ValueError(
                f"n_qubits must be 4, 8, or 10 per paper ablation; got {n_qubits}"
            )

        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self._feature_map: ZZFeatureMap | None = None

    def build(self) -> ZZFeatureMap:
        """
        Construct and return the ZZFeatureMap circuit.

        Returns:
            Configured Qiskit ZZFeatureMap with n_qubits, reps, entanglement.
        """
        # Section III-A, Eq. 3: U_phi(x)|0>^n
        self._feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.reps,               # ASSUMED: 2 (confidence 0.60)
            entanglement=self.entanglement, # ASSUMED: 'full' (confidence 0.60)
        )
        logger.debug(
            "Built ZZFeatureMap: n_qubits=%d, reps=%d, entanglement=%s",
            self.n_qubits, self.reps, self.entanglement,
        )
        return self._feature_map

    def get_circuit(self) -> QuantumCircuit:
        """
        Return the compiled QuantumCircuit for inspection or circuit drawing.

        Returns:
            QuantumCircuit representing U_phi.
        """
        fm = self.build()
        return fm.decompose()

    def __repr__(self) -> str:
        return (
            f"QSVMFeatureMap(n_qubits={self.n_qubits}, "
            f"reps={self.reps}, entanglement={self.entanglement!r})"
        )
