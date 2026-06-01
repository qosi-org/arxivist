"""
quantum_kernel.py — Quantum Kernel Estimator

Implements K(x, z) = |⟨Φ(x)|Φ(z)⟩|²  [EQ3]

Two estimation modes:
  1. Exact (use_statevector=True, default for simulation):
       Computes |⟨sv_x, sv_z⟩|² directly from Aer statevectors.
       Zero shot noise — matches the paper's "ideal" (dashed) curves.

  2. Shot-based (use_statevector=False, hardware emulation):
       Applies U†_Φ(x) U_Φ(z) to |0^n⟩, counts all-zero outcomes.
       Sampling error ~ O(1/√shots) per entry [EQ13].
       Paper used 50,000 shots per kernel entry on hardware.

Kernel matrix is symmetric: only upper triangle computed, then reflected.
PSD enforcement clips negative eigenvalues to ε (required before sklearn SVM).

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
Section: "Quantum kernel estimation"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from tqdm import tqdm

from qsvm.feature_map import FeatureMap


class QuantumKernelEstimator:
    """
    Estimates the quantum kernel K(x, z) = |⟨Φ(x)|Φ(z)⟩|²  [EQ3].

    Parameters
    ----------
    feature_map : FeatureMap
        Feature map instance defining the encoding circuit.
    backend : AerSimulator
        Qiskit Aer backend (statevector or qasm).
    shots : int
        Shots per kernel entry (only used when use_statevector=False).
        Paper uses 50,000; default is 1024 for fast simulation.
    use_statevector : bool
        If True (default), compute kernel exactly via statevector inner product.
        If False, use shot-based estimation (hardware emulation mode).
    """

    def __init__(
        self,
        feature_map: FeatureMap,
        backend: Optional[AerSimulator] = None,
        shots: int = 1024,
        use_statevector: bool = True,
    ) -> None:
        self.feature_map = feature_map
        self.shots = shots
        self.use_statevector = use_statevector

        if backend is None:
            if use_statevector:
                self.backend = AerSimulator(method="statevector")
            else:
                self.backend = AerSimulator(method="automatic")
        else:
            self.backend = backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        Compute K(x, z) = |⟨Φ(x)|Φ(z)⟩|²  [EQ3].

        Parameters
        ----------
        x, z : np.ndarray, shape [n_qubits]

        Returns
        -------
        float in [0, 1]
        """
        assert x.shape == (self.feature_map.n_qubits,), (
            f"Expected x shape [{self.feature_map.n_qubits}], got {x.shape}"
        )
        assert z.shape == (self.feature_map.n_qubits,), (
            f"Expected z shape [{self.feature_map.n_qubits}], got {z.shape}"
        )

        if self.use_statevector:
            return self._evaluate_statevector(x, z)
        else:
            return self._evaluate_shots(x, z)

    def build_kernel_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Compute full kernel matrix K[i,j] = evaluate(X[i], Y[j]).

        If Y is None, compute the symmetric training kernel (upper triangle only,
        then reflected). This gives exact symmetry K[i,j] == K[j,i].

        Sampling complexity: O(ε^{-2} |T|^4) total shots for ‖K−K̂‖ ≤ ε  [EQ13].

        Parameters
        ----------
        X : np.ndarray, shape [N, n_qubits]
        Y : np.ndarray, shape [M, n_qubits], optional
            If None, computes symmetric X×X kernel.
        verbose : bool
            Show tqdm progress bar.

        Returns
        -------
        np.ndarray, shape [N, N] or [N, M], dtype float64
        """
        assert X.ndim == 2 and X.shape[1] == self.feature_map.n_qubits, (
            f"Expected X shape [N, {self.feature_map.n_qubits}], got {X.shape}"
        )

        symmetric = Y is None
        if symmetric:
            Y = X

        N, M = len(X), len(Y)
        K = np.zeros((N, M), dtype=float)

        total = (N * (N + 1)) // 2 if symmetric else N * M
        desc = "Building kernel matrix"

        with tqdm(total=total, desc=desc, disable=not verbose) as pbar:
            for i in range(N):
                j_start = i if symmetric else 0
                for j in range(j_start, M):
                    val = self.evaluate(X[i], Y[j])
                    K[i, j] = val
                    if symmetric and i != j:
                        K[j, i] = val   # reflect upper triangle
                    pbar.update(1)

        # Diagonal of symmetric kernel must be 1.0 exactly
        if symmetric:
            np.fill_diagonal(K, 1.0)

        return K

    def enforce_psd(self, K: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Enforce positive semi-definiteness by clipping negative eigenvalues.

        Shot noise can produce slightly negative eigenvalues; sklearn SVM
        requires a PSD kernel matrix. Paper mentions this issue (Supp. §QKE).

        Parameters
        ----------
        K : np.ndarray, shape [N, N], symmetric
        epsilon : float
            Minimum eigenvalue floor.

        Returns
        -------
        np.ndarray, shape [N, N], PSD
        """
        assert K.ndim == 2 and K.shape[0] == K.shape[1], (
            f"K must be square, got {K.shape}"
        )
        # Symmetrise numerically (guard against floating-point asymmetry)
        K_sym = (K + K.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(K_sym)
        eigvals_clipped = np.maximum(eigvals, epsilon)
        K_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        # Restore exact 1.0 diagonal
        np.fill_diagonal(K_psd, 1.0)
        return K_psd

    # ------------------------------------------------------------------
    # Private: exact statevector mode
    # ------------------------------------------------------------------

    def _evaluate_statevector(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        Exact kernel via statevector inner product.

        K(x,z) = |⟨Φ(x)|Φ(z)⟩|²  [EQ3]

        Tensor flow:
            sv_x: [2^n] complex128 ← feature_map.get_statevector(x)
            sv_z: [2^n] complex128 ← feature_map.get_statevector(z)
            overlap = conj(sv_x) @ sv_z  → complex scalar
            K = |overlap|² → float64
        """
        sv_x = self.feature_map.get_statevector(x, self.backend)  # [2^n] complex
        sv_z = self.feature_map.get_statevector(z, self.backend)  # [2^n] complex

        # Inner product ⟨Φ(x)|Φ(z)⟩  [EQ3]
        overlap = np.vdot(sv_x, sv_z)   # conjugates first argument
        K_val = float(np.abs(overlap) ** 2)

        # Numerical safety: clamp to [0, 1]
        return float(np.clip(K_val, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Private: shot-based mode (hardware emulation)
    # ------------------------------------------------------------------

    def _evaluate_shots(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        Shot-based kernel estimation via transition amplitude measurement.

        Circuit: U†_Φ(x) U_Φ(z) applied to |0^n⟩; count '0...0' outcomes.
        Frequency ν_{0...0} → K(x,z) with sampling error O(shots^{-1/2}).

        Tensor flow:
            circuit = [|0^n⟩] → U_Phi(z) → U†_Phi(x) → measure all
            K = counts['00...0'] / shots
        """
        qc = self._build_kernel_circuit(x, z)
        job = self.backend.run(qc, shots=self.shots)
        counts = job.result().get_counts(qc)

        # All-zero bitstring '00...0'  [Supp. EQ, "Quantum kernel estimation"]
        zero_key = "0" * self.feature_map.n_qubits
        K_val = counts.get(zero_key, 0) / self.shots
        return float(np.clip(K_val, 0.0, 1.0))

    def _build_kernel_circuit(
        self, x: np.ndarray, z: np.ndarray
    ) -> QuantumCircuit:
        """
        Compose U†_Φ(x) after U_Φ(z), prepend initialisation, append measurement.

        K(x,z) = |⟨0^n|U†_Φ(x) U_Φ(z)|0^n⟩|²  [EQ3]
        """
        n = self.feature_map.n_qubits
        qc = QuantumCircuit(n, n, name=f"K({x.round(2)},{z.round(2)})")

        # U_Phi(z)
        phi_z = self.feature_map.get_circuit(z)
        qc.compose(phi_z, inplace=True)

        # U†_Phi(x)
        phi_x_inv = self.feature_map.get_inverse_circuit(x)
        qc.compose(phi_x_inv, inplace=True)

        # Measure all qubits
        qc.measure(range(n), range(n))
        return qc

    def __repr__(self) -> str:
        mode = "statevector" if self.use_statevector else f"shots={self.shots}"
        return (
            f"QuantumKernelEstimator("
            f"feature_map={self.feature_map!r}, mode={mode})"
        )
