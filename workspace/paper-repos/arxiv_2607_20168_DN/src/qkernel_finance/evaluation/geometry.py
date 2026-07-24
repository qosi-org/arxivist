"""
Regularized geometric difference g(Kc || Kq) (Huang et al. 2021), Sec 4.5, Eq. (2).

    g(Kc || Kq) = sqrt( || sqrt(Kq) (Kc + lambda_g * N * I)^-1 sqrt(Kq) ||_inf )

A necessary-but-not-sufficient condition for quantum advantage: measures how
geometrically distinct the quantum and (tuned) classical Gram matrices are.
The paper's key finding (Sec 8) is that g is large throughout (g >> 1) but
uncorrelated with realized out-of-sample gains (rho=-0.20) -- i.e. geometric
difference alone does not predict where quantum kernels help.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


class GeometricDifference:
    """Computes the regularized geometric difference between a classical and quantum Gram matrix."""

    def compute(self, K_classical: np.ndarray, K_quantum: np.ndarray, lambda_g: float = 1e-6) -> float:
        """
        Args:
            K_classical: [M, M] classical (e.g. tuned RBF) Gram matrix, computed
                on a subset (400 points per window, per Sec 4.5).
            K_quantum: [M, M] quantum Gram matrix on the same subset.
            lambda_g: regularization strength (1e-6, per Eq. 2).

        Returns:
            Scalar g >= 0.
        """
        m = K_classical.shape[0]
        assert K_classical.shape == (m, m) == K_quantum.shape

        sqrt_Kq = sqrtm(K_quantum + 1e-10 * np.eye(m)).real  # small jitter for numerical PSD stability
        regularized_Kc = K_classical + lambda_g * m * np.eye(m)
        inner = sqrt_Kq @ np.linalg.inv(regularized_Kc) @ sqrt_Kq
        inf_norm = np.max(np.sum(np.abs(inner), axis=1))  # matrix infinity norm (max abs row sum)
        return float(np.sqrt(max(inf_norm, 0.0)))

    def __repr__(self) -> str:  # noqa: D105
        return "GeometricDifference()"
