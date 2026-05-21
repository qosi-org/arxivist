"""Compact Swap Test implementation (Algorithm 4) using Qiskit 0.45.3 APIs.

This module implements a compact swap test circuit and a `CompactSwapTest`
wrapper which supports two execution modes:
 - statevector (exact probabilities) via Aer `statevector_simulator`
 - shot-based (sampling) via Aer `qasm_simulator`

IMPORTANT: This code uses the legacy Qiskit 0.x style APIs (execute(), result.get_statevector(), etc.)
to stay compatible with the project's pinned Qiskit 0.45.3.
"""
from __future__ import annotations

from typing import Tuple, Dict
import math
import logging

import numpy as np
from qiskit import QuantumCircuit, Aer, execute

logger = logging.getLogger(__name__)


class CompactSwapTest:
    """Compact swap test wrapper.

    Methods
    -------
    run(psi, phi) -> (swap_test_probability, angular_distance)
        Build and execute circuit, returning the scalar probability and angular distance (radians).
    build_circuit(psi, phi) -> QuantumCircuit
        Construct the Qiskit circuit implementing the compact swap test per plan pseudocode.
    compute_probability(counts, shots) -> float
        Compute p0 and p1 from counts and return swap_test_probability (1 - 2*p0 + p1).
    """

    def __init__(self, use_statevector: bool = True, shots: int = 1024):
        self.use_statevector = bool(use_statevector)
        self.shots = int(shots)

    def build_circuit(self, psi: np.ndarray, phi: np.ndarray) -> QuantumCircuit:
        """Construct the compact swap test QuantumCircuit.

        Circuit layout (as specified in the plan):
          - ancilla qubit at index 0 (q_anc)
          - data qubits at indices 1..(n_qubits-1)
          - initialize data register with `psi` (length must be power-of-two)
          - apply Pauli-X to data qubit 1 (q2[1]) if present (per pseudocode)
          - Hadamard on ancilla
          - Controlled-SWAP(ancilla, data_q0, data_q1)
          - Hadamard on ancilla
          - Measure ancilla

        Notes:
          - psi must be a unit vector with length 2^m for integer m >= 1.
          - phi is not explicitly re-initialized into data register here because
            ALGO3_prep constructs psi such that the required amplitudes are placed
            in the data register; phi is retained for bookkeeping and potential checks.
        """
        psi = np.asarray(psi, dtype=complex).ravel()
        phi = np.asarray(phi, dtype=complex).ravel()

        L = psi.size
        if L == 0 or (L & (L - 1)) != 0:
            raise ValueError("psi length must be a power of two and non-zero")

        n_data_qubits = int(round(math.log2(L)))
        total_qubits = 1 + n_data_qubits  # ancilla + data qubits

        qc = QuantumCircuit(total_qubits, 1)

        # Initialize the data register (qubits 1..total_qubits-1) with psi
        data_qubits = list(range(1, total_qubits))
        # Qiskit's initialize expects the qubit ordering list matching the statevector endianness
        qc.initialize(psi, data_qubits)

        # Apply Pauli-X to q2[1] (which is data_qubits[1] if exists)
        if len(data_qubits) >= 2:
            qc.x(data_qubits[1])

        # Hadamard to ancilla (qubit 0)
        qc.h(0)

        # Controlled-SWAP using the legacy QuantumCircuit.cswap API.
        # If there is only one data qubit, leave the circuit as-is.
        if len(data_qubits) >= 2:
            qc.cswap(0, data_qubits[0], data_qubits[1])
        else:
            # No-op for single data qubit case; leave circuit structure intact.
            pass

        # Hadamard on ancilla again
        qc.h(0)

        # Measure ancilla into classical bit 0
        qc.measure(0, 0)

        return qc

    @staticmethod
    def _statevector_probabilities_from_result(result, circuit) -> Tuple[float, float]:
        """Extract ancilla p0 and p1 from a statevector result.

        Uses the legacy `result.get_statevector(circuit)` or `result.get_statevector()`
        return and assumes ancilla qubit is qubit-0 (least-significant index in basis ordering).
        """
        try:
            statevector = result.get_statevector(circuit)
        except Exception:
            # fallback: some versions expose statevector without passing circuit
            statevector = result.get_statevector()

        sv = np.asarray(statevector, dtype=complex).ravel()
        probs = np.abs(sv) ** 2
        n = sv.size
        # ancilla is qubit 0 (least significant); so indices where (idx & 1)==0 correspond to ancilla=0
        idxs = np.arange(n, dtype=int)
        ancilla_bits = idxs & 1  # 0 => ancilla measured 0, 1 => ancilla measured 1
        p0 = float(np.sum(probs[ancilla_bits == 0]))
        p1 = float(np.sum(probs[ancilla_bits == 1]))
        return p0, p1

    @staticmethod
    def compute_probability_from_counts(counts: Dict[str, int], shots: int) -> Tuple[float, float]:
        """Compute p0 and p1 from qasm counts.

        Qiskit `get_counts` returns keys as bitstrings with qubit ordering matching the circuit.
        For legacy execute() with single measured ancilla at position 0 the bitstring length equals 1,
        but when other classical bits present the key ordering is 'c{n-1}...c0'.
        We interpret keys accordingly.
        """
        total = float(shots)
        # counts keys are strings like '0' or '1' (single-bit measurement)
        counts0 = 0
        counts1 = 0
        for k, v in counts.items():
            bit = k.strip()
            if len(bit) == 0:
                continue
            # For single-bit measurement, bit is '0' or '1'
            if bit[-1] == "0":
                counts0 += v
            else:
                counts1 += v
        p0 = counts0 / total
        p1 = counts1 / total
        return float(p0), float(p1)

    def run(self, psi: np.ndarray, phi: np.ndarray) -> Tuple[float, float]:
        """Execute the compact swap test and return (swap_test_probability, angular_distance).

        If `use_statevector` is True the circuit is executed on the Aer statevector_simulator
        and exact probabilities are derived from the statevector. Otherwise the qasm_simulator
        is used with `shots` and swap_test_probability is estimated from counts.

        Returns:
            (swap_test_probability, angular_distance_in_radians)
        """
        psi = np.asarray(psi, dtype=float).ravel()
        phi = np.asarray(phi, dtype=float).ravel()

        qc = self.build_circuit(psi, phi)

        if self.use_statevector:
            backend = Aer.get_backend("statevector_simulator")
            job = execute(qc, backend)
            result = job.result()
            p0, p1 = self._statevector_probabilities_from_result(result, qc)
        else:
            backend = Aer.get_backend("qasm_simulator")
            job = execute(qc, backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(qc)
            p0, p1 = self.compute_probability_from_counts(counts, self.shots)

        # Compute swap_test_probability per EQ3: swap_test_probability = 1 - 2*p0 + p1
        swap_test_probability = 1.0 - 2.0 * p0 + p1

        # Numerically clamp to [0,1]
        swap_test_probability = max(0.0, min(1.0, float(swap_test_probability)))

        # Angular distance per EQ19: 2 * arccos(sqrt(swap_test_probability))
        # Ensure numeric stability of sqrt argument
        sq = max(0.0, min(1.0, swap_test_probability))
        angular_distance = 2.0 * math.acos(math.sqrt(sq))

        return float(swap_test_probability), float(angular_distance)
