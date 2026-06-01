"""
feature_map.py — Quantum Feature Map

Implements the parameterised feature map circuit from Havlicek et al. (2018),
Section "Quantum feature map", Eqs. 1–2 and 5.

The map encodes classical data x ∈ (0, 2π]^n into a quantum state |Φ(x)⟩ via:

    𝒰_Φ(x) = U_Φ(x) H^⊗n U_Φ(x) H^⊗n        [EQ2]

where the diagonal phase gate is:

    U_Φ(x) = exp(i Σ_{S⊆[n]} φ_S(x) Π_{i∈S} Z_i)   [EQ1]

For the 2-qubit experiment (n=2, d=2):

    φ_{i}(x)   = x_i                                  [EQ5]
    φ_{1,2}(x) = (π − x_1)(π − x_2)                  [EQ5]

Gate decompositions follow Fig. 1c of the paper:
  • Single-qubit phase: exp(i φ_i Z_i)  →  RZ(2 φ_i)
  • Two-qubit ZZ:       exp(i φ_{ij} Z_i Z_j)  →  CNOT – RZ(2 φ_{ij}) – CNOT

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, FrozenSet, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class FeatureMap:
    """
    Constructs the quantum feature map circuit 𝒰_Φ(x) [EQ2].

    The circuit is built without final measurements so it can be:
      - executed with a statevector backend (exact simulation)
      - composed with its inverse for kernel estimation (EQ3)
      - composed with variational circuit W(θ) for QVC

    Parameters
    ----------
    n_qubits : int
        Number of qubits n. Paper uses n=2.
    reps : int
        Number of (H^⊗n → U_Φ) repetition pairs. Paper uses reps=2,
        giving the structure: H^⊗n U_Φ(x) H^⊗n U_Φ(x)  (right-to-left).
        Must be 2 to match the paper.
    """

    def __init__(self, n_qubits: int = 2, reps: int = 2) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if reps != 2:
            raise ValueError(
                f"reps must be 2 (paper uses double U_Phi structure), got {reps}"
            )
        self.n_qubits = n_qubits
        self.reps = reps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def phi_coefficients(self, x: np.ndarray) -> Dict[FrozenSet[int], float]:
        """
        Compute the encoding coefficients φ_S(x) for all subsets S ⊆ [n].

        For the 2-qubit paper experiment [EQ5]:
            φ_{i}(x)   = x_i          (single-qubit terms)
            φ_{1,2}(x) = (π − x_1)(π − x_2)  (two-qubit cross term)

        For general n, higher-order terms φ_S with |S|>2 are set to zero
        (paper restricts to |S| ≤ 2, "Ising-type interactions").

        Parameters
        ----------
        x : np.ndarray, shape [n_qubits]
            Classical data point in (0, 2π]^n.

        Returns
        -------
        Dict mapping FrozenSet[int] → float
        """
        assert x.shape == (self.n_qubits,), (
            f"Expected x shape [{self.n_qubits}], got {x.shape}"
        )
        coeffs: Dict[FrozenSet[int], float] = {}

        # |S|=1 terms: φ_{i}(x) = x_i  [EQ5]
        for i in range(self.n_qubits):
            coeffs[frozenset([i])] = float(x[i])

        # |S|=2 terms: φ_{i,j}(x) = (π − x_i)(π − x_j)  [EQ5]
        for i, j in combinations(range(self.n_qubits), 2):
            coeffs[frozenset([i, j])] = (math.pi - float(x[i])) * (math.pi - float(x[j]))

        # |S|>2: not used (paper restricts to |S|≤2 — "Ising interactions")
        return coeffs

    def get_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Build the feature map circuit 𝒰_Φ(x) = U_Φ(x) H^⊗n U_Φ(x) H^⊗n [EQ2].

        The circuit acts on |0⟩^n. No measurements are appended.

        Parameters
        ----------
        x : np.ndarray, shape [n_qubits]

        Returns
        -------
        QuantumCircuit (n_qubits wide, no classical register)
        """
        assert x.shape == (self.n_qubits,), (
            f"Expected x shape [{self.n_qubits}], got {x.shape}"
        )
        qc = QuantumCircuit(self.n_qubits, name=f"Phi({x.round(3)})")
        coeffs = self.phi_coefficients(x)

        # Build right-to-left: first H^⊗n, then U_Phi, then H^⊗n, then U_Phi
        # (Qiskit appends left-to-right so this matches the mathematical order)
        for _ in range(self.reps):
            self._append_hadamard_layer(qc)
            self._append_u_phi(qc, coeffs)

        return qc

    def get_inverse_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Return 𝒰_Φ(x)^† = (U_Φ H^⊗n U_Φ H^⊗n)^†.

        Used in kernel circuit: U†_Φ(x) U_Φ(z) for K(x,z) estimation [EQ3].

        Parameters
        ----------
        x : np.ndarray, shape [n_qubits]

        Returns
        -------
        QuantumCircuit (inverse of get_circuit(x))
        """
        return self.get_circuit(x).inverse()

    def get_statevector(
        self,
        x: np.ndarray,
        backend: Optional[AerSimulator] = None,
    ) -> np.ndarray:
        """
        Execute the feature map circuit and return the statevector |Φ(x)⟩.

        Parameters
        ----------
        x : np.ndarray, shape [n_qubits]
        backend : AerSimulator (statevector mode)
            If None, creates a local statevector_simulator.

        Returns
        -------
        np.ndarray, shape [2^n_qubits], dtype complex128
            Normalised statevector |Φ(x)⟩.
        """
        from qiskit_aer import AerSimulator

        if backend is None:
            backend = AerSimulator(method="statevector")

        qc = self.get_circuit(x)
        qc.save_statevector()

        job = backend.run(qc)
        result = job.result()
        sv = np.array(result.get_statevector(qc), dtype=complex)
        return sv

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _append_hadamard_layer(self, qc: QuantumCircuit) -> None:
        """Append H^⊗n (Hadamard on every qubit)."""
        for i in range(self.n_qubits):
            qc.h(i)

    def _append_u_phi(
        self,
        qc: QuantumCircuit,
        coeffs: Dict[FrozenSet[int], float],
    ) -> None:
        """
        Append U_Φ(x) = exp(i Σ_S φ_S Π_{i∈S} Z_i) [EQ1].

        Gate decompositions (Fig. 1c of paper):
          |S|=1: exp(i φ_i Z_i)     →  RZ(2 φ_i) on qubit i
          |S|=2: exp(i φ_{ij} Z_i Z_j)  →  CNOT(i→j) – RZ(2 φ_{ij}, j) – CNOT(i→j)
        """
        # Single-qubit phase gates [EQ1, |S|=1]
        for subset, phi in coeffs.items():
            if len(subset) == 1:
                (i,) = subset
                # exp(i φ Z) = RZ(2φ) up to global phase
                qc.rz(2.0 * phi, i)

        # Two-qubit ZZ coupling [EQ1, |S|=2]; Fig. 1c decomposition
        for subset, phi in coeffs.items():
            if len(subset) == 2:
                i, j = sorted(subset)
                # exp(i φ Z_i Z_j):  CNOT(i,j) – RZ(2φ, j) – CNOT(i,j)
                qc.cx(i, j)
                qc.rz(2.0 * phi, j)
                qc.cx(i, j)

    def __repr__(self) -> str:
        return f"FeatureMap(n_qubits={self.n_qubits}, reps={self.reps})"
