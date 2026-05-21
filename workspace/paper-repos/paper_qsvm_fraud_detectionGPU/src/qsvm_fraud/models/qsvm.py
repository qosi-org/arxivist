"""
models/qsvm.py — Quantum Support Vector Machine (primary model).

Implements the full QSVM pipeline:
  1. Encode training data via ZZFeatureMap into quantum states |phi(x)>
  2. Compute quantum kernel matrix K_ij = |<phi(xi)|phi(xj)>|^2  (Eq. 8)
  3. Fit classical SVC(kernel='precomputed') on kernel matrix      (Section III-C-2)
  4. At inference: compute K_test and apply decision function      (Eq. 9)

Paper reference: Section III — Quantum Support Vector Machine
  Eq. 8: K_ij = K(xi, xj) = |<phi(xi)|phi(xj)>|^2
  Eq. 9: f(x) = sign( sum_i alpha_i * y_i * K(xi, x) + b )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import joblib
import numpy as np
from sklearn.svm import SVC

from qsvm_fraud.models.feature_map import QSVMFeatureMap
from qsvm_fraud.models.quantum_kernel import QSVMKernelComputer

logger = logging.getLogger(__name__)


class QSVM:
    """
    Quantum Support Vector Machine for binary fraud classification.

    Combines ZZFeatureMap-based quantum feature encoding with a classical SVM
    solver operating on the precomputed quantum kernel matrix.

    Training flow:
        X_train → ZZFeatureMap → K_train [N×N] → SVC(precomputed) → alphas, b

    Inference flow (Eq. 9):
        x → |phi(x)> → K(x, x_sv) → sign(Σ alpha_i * y_i * K_i + b)

    Paper: Section III, Eqs. 8–9.

    Args:
        n_qubits:     Number of qubits (== feature dimension). Paper: 4, 8, or 10.
        reps:         ZZFeatureMap repetitions. ASSUMED=2 (confidence: 0.60).
        entanglement: ZZFeatureMap entanglement. ASSUMED='full' (confidence: 0.60).
        C:            SVM regularization. ASSUMED=1.0 (confidence: 0.65).
        backend:      Qiskit backend. ASSUMED='statevector_simulator' (confidence: 0.85).
        cache_kernel: Cache K_train.npy to avoid recomputation on re-runs.
        probability:  Enable predict_proba via Platt scaling (needed for AUC).
        random_state: Reproducibility seed.
        cache_dir:    Directory for kernel cache and model checkpoint.
    """

    def __init__(
        self,
        n_qubits: int = 10,
        reps: int = 2,                        # ASSUMED (confidence: 0.60)
        entanglement: str = "full",           # ASSUMED (confidence: 0.60)
        C: float = 1.0,                       # ASSUMED (confidence: 0.65)
        backend: str = "statevector_simulator", # ASSUMED (confidence: 0.85)
        cache_kernel: bool = True,
        probability: bool = True,
        random_state: int = 42,
        cache_dir: str = "checkpoints/",
    ) -> None:
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.backend = backend
        self.cache_kernel = cache_kernel
        self.probability = probability
        self.random_state = random_state
        self.cache_dir = Path(cache_dir)

        self._feature_map = QSVMFeatureMap(
            n_qubits=n_qubits,
            reps=reps,
            entanglement=entanglement,
        )
        self._kernel_computer = QSVMKernelComputer(
            feature_map=self._feature_map,
            backend=backend,
            cache_dir=str(self.cache_dir),
        )
        # Classical SVM solver with precomputed quantum kernel (Section III-C-2)
        self._svc = SVC(
            kernel="precomputed",
            C=self.C,                         # ASSUMED: 1.0 (confidence: 0.65)
            probability=self.probability,
            random_state=self.random_state,
        )

        self._X_train: Optional[np.ndarray] = None
        self._is_fitted: bool = False
        self._cache_key: str = f"qsvm_{n_qubits}qubit"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QSVM":
        """
        Train the QSVM model.

        Steps:
          1. Compute quantum kernel matrix K_train [N×N]  — Eq. 8
          2. Fit SVC(kernel='precomputed') on K_train     — Section III-C-2

        Args:
            X_train: Training features [N_train, n_qubits] float64.
            y_train: Training labels [N_train] int {0, 1}.

        Returns:
            self (fitted model).
        """
        assert X_train.ndim == 2, f"X_train must be 2D; got {X_train.shape}"
        assert X_train.shape[1] == self.n_qubits, (
            f"Feature dim {X_train.shape[1]} != n_qubits {self.n_qubits}"
        )
        assert len(X_train) == len(y_train), "X_train and y_train length mismatch"

        logger.info(
            "QSVM.fit() — n_train=%d, n_qubits=%d, C=%.3f",
            len(X_train), self.n_qubits, self.C,
        )

        # Step 1: Quantum kernel matrix — Eq. 8: K_ij = |<phi(xi)|phi(xj)>|^2
        cache_key = self._cache_key if self.cache_kernel else None
        K_train = self._kernel_computer.compute_kernel_matrix(
            X_train=X_train,
            cache_key=cache_key,
        )

        # Step 2: Classical SVM dual optimisation — Section III-C-2
        logger.info("Fitting SVC(kernel='precomputed', C=%.3f)...", self.C)
        self._svc.fit(K_train, y_train)

        self._X_train = X_train.copy()
        self._is_fitted = True

        n_sv = len(self._svc.support_vectors_) if hasattr(self._svc, "support_vectors_") else "?"
        logger.info("QSVM fitted. Support vectors: %s", n_sv)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict fraud labels for test samples.

        Implements Eq. 9:
            f(x) = sign( sum_i alpha_i * y_i * K(xi, x) + b )

        Args:
            X_test: Test features [N_test, n_qubits] float64.

        Returns:
            y_pred: [N_test] int32 — binary labels {0=legitimate, 1=fraud}.
        """
        self._check_fitted()
        assert X_test.ndim == 2, f"X_test must be 2D; got {X_test.shape}"
        assert X_test.shape[1] == self.n_qubits

        # Compute K_test [N_test, N_train] — needed for Eq. 9
        cache_key = f"{self._cache_key}_test" if self.cache_kernel else None
        K_test = self._kernel_computer.compute_kernel_matrix(
            X_train=self._X_train,
            X_test=X_test,
            cache_key=cache_key,
        )

        # Decision function: Eq. 9 (applied internally by sklearn SVC)
        return self._svc.predict(K_test).astype(np.int32)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (Platt scaling on SVM scores).

        Required for AUC computation. Only available when probability=True.

        Args:
            X_test: [N_test, n_qubits] float64.

        Returns:
            proba: [N_test, 2] — columns are [P(legit), P(fraud)].
        """
        self._check_fitted()
        if not self.probability:
            raise RuntimeError(
                "predict_proba requires QSVM(probability=True). Re-initialise and refit."
            )

        K_test = self._kernel_computer.compute_kernel_matrix(
            X_train=self._X_train,
            X_test=X_test,
        )
        return self._svc.predict_proba(K_test)

    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        """
        Raw SVM decision scores (before sign). Larger = more fraud-like.

        Args:
            X_test: [N_test, n_qubits] float64.

        Returns:
            scores: [N_test] float64.
        """
        self._check_fitted()
        K_test = self._kernel_computer.compute_kernel_matrix(
            X_train=self._X_train,
            X_test=X_test,
        )
        return self._svc.decision_function(K_test)

    def save(self, path: str) -> None:
        """Serialise fitted QSVM to disk via joblib."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("QSVM model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "QSVM":
        """Load a previously saved QSVM from disk."""
        model = joblib.load(path)
        logger.info("QSVM model loaded from %s", path)
        return model

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("QSVM is not fitted. Call fit() before predict().")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"QSVM(n_qubits={self.n_qubits}, reps={self.reps}, "
            f"entanglement={self.entanglement!r}, C={self.C}, "
            f"backend={self.backend!r}, status={status})"
        )
