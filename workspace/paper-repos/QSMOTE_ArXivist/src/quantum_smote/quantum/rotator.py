"""Quantum rotation circuit for Quantum-SMOTE (Algorithm 6).

This module initializes a quantum circuit with the normalized minority
sample as a statevector, applies RX(angle) to all qubits, and extracts the
real-valued synthetic feature vector from the simulated statevector.

IMPORTANT:
- No entangling gates are used.
- `extract_synthetic` supports the `first_F` mitigation strategy from R1.
"""
from __future__ import annotations

from typing import Optional
import math
import logging

import numpy as np
from qiskit import QuantumCircuit, Aer, execute

logger = logging.getLogger(__name__)


class QuantumRotator:
    """Apply RX rotations to a minority sample and extract synthetic data."""

    def __init__(
        self,
        statevector_extraction_strategy: str = "first_F",
        backend_name: str = "statevector_simulator",
    ) -> None:
        self.statevector_extraction_strategy = statevector_extraction_strategy
        self.backend_name = backend_name

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float).ravel()
        norm = float(np.sqrt(np.sum(arr ** 2)))
        if norm <= 0.0:
            raise ValueError("Cannot normalize zero vector")
        return arr / norm

    def build_circuit(self, data_point: np.ndarray, angle: float) -> QuantumCircuit:
        """Construct the RX-only circuit for the minority sample.

        The plan specifies:
        - `n = floor(log2(F))`
        - `add = 1 if F % n != 0 else 0`
        - `loop_ctr = round(F / n + add)`
        - `n_qubits = n`
        - initialize the circuit with the normalized sample (truncated/padded
          to 2**n_qubits amplitudes)
        - apply RX(angle) to each qubit

        No entangling gates are used.
        """
        sample = np.asarray(data_point, dtype=float).ravel()
        if sample.size == 0:
            raise ValueError("data_point must be non-empty")

        # Follow the plan's qubit scaling formula.
        F = int(sample.size)
        n = int(math.floor(math.log2(F)))
        if n < 1:
            n = 1
        add = 1 if (F % n) != 0 else 0
        loop_ctr = int(round((F / float(n)) + add))
        n_qubits = n

        # n_qubits defines the register size; statevector length is 2**n_qubits.
        state_len = 2**n_qubits
        normalized = self._normalize_array(sample)

        # Truncate or pad to exactly state_len amplitudes for initialize().
        if normalized.size >= state_len:
            init_state = normalized[:state_len]
        else:
            init_state = np.zeros(state_len, dtype=float)
            init_state[: normalized.size] = normalized
            # renormalize after padding to maintain a valid statevector.
            init_state = self._normalize_array(init_state)

        qc = QuantumCircuit(n_qubits)
        qc.initialize(init_state, list(range(n_qubits)))

        for qubit_idx in range(n_qubits):
            qc.rx(angle, qubit_idx)

        return qc

    @staticmethod
    def extract_synthetic(statevector: np.ndarray, target_dim: int, strategy: str = "first_F") -> np.ndarray:
        """Extract a real-valued synthetic vector from a simulated statevector.

        Supported strategies:
        - `first_F`: return `real(statevector[:target_dim])`
        - `reshape`: reshape the real part into a flat vector and take first F

        The project mitigation for R1 requires `first_F` support.
        """
        sv = np.asarray(statevector)
        if target_dim <= 0:
            raise ValueError("target_dim must be positive")

        if strategy == "first_F":
            syn = np.real(sv[:target_dim])
        elif strategy == "reshape":
            syn = np.real(sv).reshape(-1)[:target_dim]
        else:
            raise ValueError(f"Unknown extraction strategy: {strategy}")

        if syn.size < target_dim:
            padded = np.zeros(target_dim, dtype=float)
            padded[: syn.size] = syn
            syn = padded

        return syn.astype(float)

    def rotate(self, data_point: np.ndarray, angle: float) -> np.ndarray:
        """Run the RX-only circuit and return the extracted synthetic point."""
        qc = self.build_circuit(data_point, angle)
        backend = Aer.get_backend(self.backend_name)
        job = execute(qc, backend)
        result = job.result()

        try:
            statevector = result.get_statevector(qc)
        except Exception:
            statevector = result.get_statevector()

        target_dim = int(np.asarray(data_point).ravel().size)
        return self.extract_synthetic(
            statevector=statevector,
            target_dim=target_dim,
            strategy=self.statevector_extraction_strategy,
        )
