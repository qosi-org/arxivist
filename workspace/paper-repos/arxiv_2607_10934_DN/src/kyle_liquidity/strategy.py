"""
Insider's equilibrium optimal trading strategy (eq. 3.8):

    dX*_t = sigma_t^2 (M*_t)^{-1} (Sigma*_t)^{-1} (v~ - p0 - xi*_t) dt

Since P*_t = p0 + xi*_t (eq. 3.9), we implement the drift directly in terms of the
observed price P_t rather than xi_t, which is algebraically identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InsiderStrategy:
    """Stateless equilibrium strategy: computes dX*_t/dt given current market state."""

    eig_floor: float = 1e-10  # numerical safeguard near t=T where Sigma*_t -> 0 (Section 3, eq. 3.7)

    def _safe_inv(self, A: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (A + A.T))
        eigvals_clipped = np.clip(eigvals, self.eig_floor, None)
        return eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T

    def drift(
        self,
        v_true: np.ndarray,
        p_t: np.ndarray,
        M_t: np.ndarray,
        Sigma_t: np.ndarray,
        sigma_t: np.ndarray,
    ) -> np.ndarray:
        """Eq. (3.8): dX*_t/dt = sigma_t^2 (M*_t)^{-1} (Sigma*_t)^{-1} (v~ - p_t)."""
        M_inv = np.linalg.inv(M_t)
        Sigma_inv = self._safe_inv(Sigma_t)
        sigma_sq = sigma_t @ sigma_t
        return sigma_sq @ M_inv @ Sigma_inv @ (v_true - p_t)

    def is_inconspicuous(
        self, drifts: np.ndarray, tol: float = 1e-2
    ) -> bool:
        """Empirical check of eq. (3.12): E[dX*_t/dt | F^M_t] = 0.

        Along a single simulated path we can only check that the SAMPLE MEAN of the
        drift is small relative to its scale -- this is a Monte-Carlo sanity check, not
        a proof of the conditional-expectation identity, which is guaranteed analytically
        by construction (Proposition D.5). See run_verification.py for a multi-path version.
        """
        mean_drift = np.mean(drifts, axis=0)
        scale = np.mean(np.abs(drifts)) + 1e-12
        return bool(np.all(np.abs(mean_drift) / scale < tol))
