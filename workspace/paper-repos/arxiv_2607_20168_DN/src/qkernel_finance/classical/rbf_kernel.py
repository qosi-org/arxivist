"""
Classical RBF kernel-swap control (Sec 4.3, krr-rbf): identical top-8 inputs,
subsample, solver, and hyperparameter budget as the quantum kernels, so that
any performance delta isolates kernel geometry.
"""
from __future__ import annotations

import numpy as np


class ClassicalRBFKernel:
    """Standard RBF kernel on the same top-8 (bandwidth-scaled) inputs as the quantum branch."""

    def compute_gram(self, X: np.ndarray, gamma: float) -> np.ndarray:
        """
        Args:
            X: [N, 8] bandwidth-scaled top-8 inputs.
            gamma: RBF bandwidth (tuned per window on the same grid as the quantum kernels).

        Returns:
            [N, N] Gram matrix.
        """
        sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
        sq_dists = np.clip(sq_dists, a_min=0.0, a_max=None)
        return np.exp(-gamma * sq_dists)

    def compute_cross_gram(self, X_a: np.ndarray, X_b: np.ndarray, gamma: float) -> np.ndarray:
        sq_dists = np.sum(X_a**2, axis=1)[:, None] + np.sum(X_b**2, axis=1)[None, :] - 2 * X_a @ X_b.T
        sq_dists = np.clip(sq_dists, a_min=0.0, a_max=None)
        return np.exp(-gamma * sq_dists)

    def __repr__(self) -> str:  # noqa: D105
        return "ClassicalRBFKernel()"
