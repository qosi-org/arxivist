"""
Numerical illustration of the matrix-valued Doob-Meyer decomposition (Appendix B,
Theorem B.3): X_t = M_t + A_t, with M a local martingale and A nondecreasing predictable.

This module does NOT re-derive the theorem (it is proved rigorously in the paper via a
symmetric/antisymmetric split plus a coordinatewise scalar Doob-Meyer argument along
e_i +/- e_j directions). Instead it provides a discrete-time empirical decomposition of a
simulated matrix submartingale path, useful as a sanity check / visualization aid.
"""
from __future__ import annotations

import numpy as np


class MatrixDoobMeyer:
    """Discrete-time (running-conditional-expectation) Doob-Meyer style decomposition.

    Given a simulated path X_0, ..., X_{T} of an n x n matrix-valued process that is a
    submartingale with respect to its own filtration, decomposes it via the classical
    discrete construction:

        A_k = sum_{j<k} ( E[X_{j+1} | F_j] - X_j )     (compensator, nondecreasing in Loewner order)
        M_k = X_k - A_k                                 (martingale part)

    Since we only have ONE simulated path (not the conditional law), we approximate
    E[X_{j+1} | F_j] by a local regression against the immediate past increment's sign
    structure is not identifiable from a single path; the paper's theorem is about the
    true underlying process, not an estimator. Accordingly this implementation is
    intentionally restricted to the case where the user supplies the TRUE one-step
    conditional-mean map `cond_mean_fn(X_j, j) -> E[X_{j+1} | F_j]` (e.g. known analytically
    for a simulated model), which is the practically useful case for verifying the theorem
    against a model where the answer is already known.
    """

    def __init__(self, cond_mean_fn):
        self.cond_mean_fn = cond_mean_fn

    def decompose(self, X_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            X_paths: [T+1, n, n] array, a single realized path of the submartingale.
        Returns:
            (M_paths, A_paths), each [T+1, n, n], with X = M + A, A_0 = 0, A nondecreasing
            (in Loewner order) along the path, M_paths - A_paths reconstructing X exactly.
        """
        T_plus_1, n, _ = X_paths.shape
        A = np.zeros_like(X_paths)
        for j in range(T_plus_1 - 1):
            cond_mean_next = self.cond_mean_fn(X_paths[j], j)
            increment = cond_mean_next - X_paths[j]
            # Symmetrize the increment so A stays in the symmetric matrices, consistent
            # with the proof's reduction to the symmetric part Z=(X+X^T)/2 (Theorem B.3 proof).
            increment_sym = 0.5 * (increment + increment.T)
            # Project onto S^n_+ (nondecreasing steps) by clipping negative eigenvalues,
            # since a single realized path can locally violate monotonicity in Loewner
            # order even though the THEOREM (about the true compensator) guarantees it.
            eigvals, eigvecs = np.linalg.eigh(increment_sym)
            eigvals_clipped = np.clip(eigvals, 0.0, None)
            increment_sym = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            A[j + 1] = A[j] + increment_sym
        M = X_paths - A
        return M, A
