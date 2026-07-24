"""
Fidelity and projected quantum kernels (Sec 4.2).

Fidelity kernel:    kappa_Q(x,x') = |<psi(x)|psi(x')>|^2
Projected kernel:   phi_q(x) = (<X_q>,<Y_q>,<Z_q>);  phi(x) in R^24
                    kappa_P(x,x') = exp(-gamma * ||phi(x)-phi(x')||^2)
"""
from __future__ import annotations

import numpy as np

from qkernel_finance.quantum.feature_map import QuantumFeatureMap


class FidelityKernel:
    """kappa_Q(x,x') = |<psi(x)|psi(x')>|^2, batched via one inner-product matrix multiply."""

    def compute_gram(self, X: np.ndarray, feature_map: QuantumFeatureMap) -> np.ndarray:
        """
        Args:
            X: [N, num_qubits] bandwidth-scaled inputs.
            feature_map: a QuantumFeatureMap instance.

        Returns:
            [N, N] real-valued Gram matrix, entries in [0, 1].
        """
        psi = feature_map.states_batch(X)  # [N, 2**n] complex
        overlaps = psi @ psi.conj().T  # [N, N] complex inner products
        gram = np.abs(overlaps) ** 2
        return gram.real

    def compute_cross_gram(self, X_a: np.ndarray, X_b: np.ndarray, feature_map: QuantumFeatureMap) -> np.ndarray:
        """Cross Gram matrix between two (possibly different-sized) batches, e.g. test-vs-train."""
        psi_a = feature_map.states_batch(X_a)
        psi_b = feature_map.states_batch(X_b)
        overlaps = psi_a @ psi_b.conj().T
        return (np.abs(overlaps) ** 2).real

    def __repr__(self) -> str:  # noqa: D105
        return "FidelityKernel()"


class ProjectedQuantumKernel:
    """kappa_P(x,x') = exp(-gamma * ||phi(x)-phi(x')||^2), phi = per-qubit Bloch vectors (Huang et al. 2021)."""

    def bloch_features(self, X: np.ndarray, feature_map: QuantumFeatureMap) -> np.ndarray:
        """
        Args:
            X: [N, num_qubits] bandwidth-scaled inputs.
            feature_map: a QuantumFeatureMap instance.

        Returns:
            [N, 3*num_qubits] Bloch feature vectors (24-dim for n=8).
        """
        return np.stack([feature_map.bloch_vectors(row) for row in X], axis=0)

    def compute_gram(self, phi: np.ndarray, gamma: float) -> np.ndarray:
        """
        Args:
            phi: [N, 3*num_qubits] Bloch feature vectors.
            gamma: RBF bandwidth. ASSUMED tuned per window via the same
                validation-IC grid search as lambda (SIR ambiguities[1],
                confidence 0.5 -- paper does not state gamma's tuning procedure).

        Returns:
            [N, N] Gram matrix.
        """
        sq_dists = np.sum(phi**2, axis=1)[:, None] + np.sum(phi**2, axis=1)[None, :] - 2 * phi @ phi.T
        sq_dists = np.clip(sq_dists, a_min=0.0, a_max=None)
        return np.exp(-gamma * sq_dists)

    def compute_cross_gram(self, phi_a: np.ndarray, phi_b: np.ndarray, gamma: float) -> np.ndarray:
        sq_dists = np.sum(phi_a**2, axis=1)[:, None] + np.sum(phi_b**2, axis=1)[None, :] - 2 * phi_a @ phi_b.T
        sq_dists = np.clip(sq_dists, a_min=0.0, a_max=None)
        return np.exp(-gamma * sq_dists)

    def __repr__(self) -> str:  # noqa: D105
        return "ProjectedQuantumKernel()"
