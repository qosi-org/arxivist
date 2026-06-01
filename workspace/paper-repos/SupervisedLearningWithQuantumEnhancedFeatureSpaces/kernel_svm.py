"""
kernel_svm.py — Quantum Kernel SVM (Protocol 2)

Implements Protocol 2 from Havlicek et al. (2018), Section "Quantum kernel
estimation". The quantum computer estimates K(x_i, x_j); a classical SVM
finds the optimal separating hyperplane in kernel space.

SVM Wolfe dual objective [EQ11]:
    L_D(α) = Σ_i α_i − ½ Σ_{i,j} y_i y_j α_i α_j K(x_i, x_j)
    s.t.  Σ_i α_i y_i = 0,  α_i ≥ 0

Classification rule [EQ12]:
    ỹ(s) = sign( Σ_{i∈N_S} y_i α*_i K(x_i, s) + b )

Support vectors N_S: indices where α*_i > 0 (complementary slackness).
Bias b computed from KKT conditions.

Implementation uses sklearn.svm.SVC with kernel='precomputed' which directly
solves the Wolfe dual given the precomputed kernel matrix.

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.svm import SVC

from qsvm.quantum_kernel import QuantumKernelEstimator


class QuantumKernelSVM:
    """
    Quantum Kernel SVM — Protocol 2 of Havlicek et al. (2018).

    Wraps sklearn.svm.SVC(kernel='precomputed') with a quantum kernel matrix.

    Parameters
    ----------
    kernel_estimator : QuantumKernelEstimator
        Computes K(x_i, x_j) entries on quantum hardware / simulator.
    C : float
        SVM regularisation parameter.
        ASSUMED: C=1.0 (hard-margin SVM implied by perfectly separable data).
        conf=0.75 — adjust if training fails to converge.
    """

    def __init__(
        self,
        kernel_estimator: QuantumKernelEstimator,
        C: float = 1.0,
    ) -> None:
        self.kernel_estimator = kernel_estimator
        self.C = C
        self._svm: Optional[SVC] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._K_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> None:
        self._y_train = y_train.copy()   # store labels for get_support_vectors
        """
        Build quantum kernel matrix then solve SVM Wolfe dual  [EQ11].

        Step 1: Compute K_train[i,j] = K(x_i, x_j) via quantum circuit.
        Step 2: Enforce K_train is PSD (clip negative eigenvalues).
        Step 3: sklearn SVC.fit(K_train, y_train) solves Wolfe dual.

        Parameters
        ----------
        X_train : np.ndarray, shape [N, n_qubits]
        y_train : np.ndarray, shape [N], values in {+1, -1}
        verbose : bool
        """
        assert X_train.ndim == 2, f"X_train must be 2D, got shape {X_train.shape}"
        assert y_train.ndim == 1, f"y_train must be 1D, got shape {y_train.shape}"
        assert len(X_train) == len(y_train), "X_train and y_train must have same length"

        self._X_train = X_train.copy()

        # Step 1: Estimate quantum kernel matrix  [Supp. "Quantum kernel estimation"]
        if verbose:
            print(f"Computing {len(X_train)}×{len(X_train)} quantum kernel matrix…")
        K_train = self.kernel_estimator.build_kernel_matrix(
            X_train, verbose=verbose
        )

        # Step 2: Enforce positive semi-definiteness  [Supp. §QKE]
        psd_eps = getattr(self.kernel_estimator, '_psd_epsilon', 1e-10)
        K_train = self.kernel_estimator.enforce_psd(K_train, epsilon=psd_eps)
        self._K_train = K_train

        # Step 3: Classical SVM on precomputed quantum kernel  [EQ11]
        # sklearn SVC solves the Wolfe dual internally with LIBSVM
        self._svm = SVC(kernel="precomputed", C=self.C)
        self._svm.fit(K_train, y_train)

        if verbose:
            n_sv = len(self._svm.support_)
            print(f"Training done. Support vectors: {n_sv}/{len(X_train)}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Classify test points using trained SVM  [EQ12]:
            ỹ(s) = sign( Σ_{i∈N_S} y_i α*_i K(x_i, s) + b )

        Parameters
        ----------
        X_test : np.ndarray, shape [M, n_qubits]
        verbose : bool

        Returns
        -------
        np.ndarray, shape [M], dtype int, values in {+1, -1}
        """
        self._check_fitted()
        assert X_test.ndim == 2, f"X_test must be 2D, got {X_test.shape}"

        # Compute K_test[i,j] = K(X_test[i], X_train[j])
        # sklearn expects shape [n_test, n_train_total] (not just support vectors)
        if verbose:
            print(f"Computing test kernel matrix [{len(X_test)}×{len(self._X_train)}]…")
        K_test = self.kernel_estimator.build_kernel_matrix(
            X_test, self._X_train, verbose=verbose
        )

        return self._svm.predict(K_test).astype(int)

    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = False,
    ) -> float:
        """
        Classification success rate  [Paper: "Classification success rate"].

        Parameters
        ----------
        X_test : np.ndarray, shape [M, n_qubits]
        y_test : np.ndarray, shape [M]

        Returns
        -------
        float in [0, 1]
        """
        y_pred = self.predict(X_test, verbose=verbose)
        return float(np.mean(y_pred == y_test))

    def decision_function_values(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return raw SVM decision values Σ_i y_i α*_i K(x_i, s) + b  [EQ12].

        Used to reproduce the bottom panel of Fig. 3b in the paper, which
        plots these values for each test point.

        Parameters
        ----------
        X_test : np.ndarray, shape [M, n_qubits]

        Returns
        -------
        np.ndarray, shape [M], float64
        """
        self._check_fitted()
        K_test = self.kernel_estimator.build_kernel_matrix(
            X_test, self._X_train, verbose=False
        )
        return self._svm.decision_function(K_test)

    # ------------------------------------------------------------------
    # Support vector inspection
    # ------------------------------------------------------------------

    def get_support_vectors(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return support vectors, dual coefficients, and labels.

        Matches the format of Table S2 in the paper supplementary material:
            SV (coordinates), α_i, y_i

        Returns
        -------
        support_vectors : np.ndarray, shape [N_sv, n_qubits]
        dual_coefs_alpha : np.ndarray, shape [N_sv]  (α_i * y_i from sklearn)
        labels_y : np.ndarray, shape [N_sv]  ({+1, -1})
        """
        self._check_fitted()
        sv_indices = self._svm.support_
        support_vectors = self._X_train[sv_indices]

        # sklearn stores dual_coef_ = α_i * y_i per support vector [1, N_sv]
        dual_coefs_signed = self._svm.dual_coef_.ravel()

        # Labels directly from stored y_train (avoids re-predicting with precomputed kernel)
        labels_y = self._y_train[sv_indices].astype(int)

        # Recover α_i (unsigned): dual_coef = α_i * y_i  → α_i = |dual_coef|
        dual_coefs_alpha = np.abs(dual_coefs_signed)

        return support_vectors, dual_coefs_alpha, labels_y

    def get_bias(self) -> float:
        """
        Return the SVM bias b (intercept) from KKT conditions  [Supp. EQ, QKE].

        Returns
        -------
        float
        """
        self._check_fitted()
        return float(self._svm.intercept_[0])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._svm is None or self._X_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def __repr__(self) -> str:
        fitted = self._svm is not None
        sv_info = (
            f", n_support={len(self._svm.support_)}" if fitted else ""
        )
        return (
            f"QuantumKernelSVM("
            f"C={self.C}, "
            f"fitted={fitted}"
            f"{sv_info})"
        )
