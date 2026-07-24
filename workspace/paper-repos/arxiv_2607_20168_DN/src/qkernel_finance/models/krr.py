"""
Kernel ridge regression solver and Nystrom full-budget extension (Sec 4.3, 5.1).

    alpha_hat = (K + alpha*I)^-1 y

Nystrom extension (Williams & Seeger 2001, cited but not restated in the
paper -- SIR ambiguities[2], confidence 0.55): approximates a large NxN Gram
matrix using an MxM landmark Gram matrix (M=1536 here) plus the NxM cross
Gram matrix between all training points and the landmarks.
"""
from __future__ import annotations

import numpy as np


class KernelRidgeRegression:
    """Standard (non-approximate) closed-form KRR, used for the N=1536-observation kernel branch."""

    def fit(self, K_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
        """
        Args:
            K_train: [N, N] training Gram matrix.
            y_train: [N] targets (20-day forward returns).
            alpha: ridge regularization strength.

        Returns:
            [N] dual coefficients alpha_hat = (K + alpha*I)^-1 y.
        """
        n = K_train.shape[0]
        assert K_train.shape == (n, n), f"K_train must be square, got {K_train.shape}"
        assert y_train.shape == (n,), f"y_train shape {y_train.shape} must match K_train's {n} rows"
        return np.linalg.solve(K_train + alpha * np.eye(n), y_train)

    def predict(self, K_test_train: np.ndarray, alpha_hat: np.ndarray) -> np.ndarray:
        """
        Args:
            K_test_train: [N_test, N_train] cross Gram matrix.
            alpha_hat: [N_train] dual coefficients from fit().

        Returns:
            [N_test] predictions.
        """
        return K_test_train @ alpha_hat

    def __repr__(self) -> str:  # noqa: D105
        return "KernelRidgeRegression()"


class NystromKRR:
    """Nystrom-extended KRR: fits at full training budget (~37,800 obs) using a fixed landmark set.

    ASSUMED implementation (SIR ambiguities[2], confidence 0.55): standard
    Nystrom KRR as in Williams & Seeger (2001) -- approximate the full kernel
    matrix's action via its low-rank landmark decomposition, then solve a
    reduced M x M linear system (M = number of landmarks) instead of the full
    N x N system.
    """

    def fit(self, K_landmarks: np.ndarray, K_full_landmarks: np.ndarray, y_full: np.ndarray, alpha: float) -> np.ndarray:
        """
        Args:
            K_landmarks: [M, M] Gram matrix among the M=1536 landmark points.
            K_full_landmarks: [N, M] cross Gram matrix, all N~37,800 training
                points vs. the M landmarks.
            y_full: [N] targets for all N training points.
            alpha: ridge regularization strength.

        Returns:
            [M] reduced dual coefficients beta_hat, such that predictions for
            any new point x are K_x_landmarks @ beta_hat (see `predict`).
        """
        m = K_landmarks.shape[0]
        n = K_full_landmarks.shape[0]
        assert K_landmarks.shape == (m, m)
        assert K_full_landmarks.shape == (n, m)
        assert y_full.shape == (n,)

        # Standard Nystrom KRR reduced system: (K_lm^T K_lm + alpha * K_landmarks) beta = K_lm^T y
        # (Williams & Seeger 2001-style landmark regression; see module docstring.)
        A = K_full_landmarks.T @ K_full_landmarks + alpha * K_landmarks
        b = K_full_landmarks.T @ y_full
        beta_hat = np.linalg.solve(A + 1e-8 * np.eye(m), b)  # small jitter for numerical stability
        return beta_hat

    def predict(self, K_test_landmarks: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
        """
        Args:
            K_test_landmarks: [N_test, M] cross Gram matrix, test points vs. landmarks.
            beta_hat: [M] reduced coefficients from fit().

        Returns:
            [N_test] predictions.
        """
        return K_test_landmarks @ beta_hat

    def __repr__(self) -> str:  # noqa: D105
        return "NystromKRR()"
