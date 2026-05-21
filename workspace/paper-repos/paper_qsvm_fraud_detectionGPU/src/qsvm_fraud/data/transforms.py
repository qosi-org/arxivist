"""
data/transforms.py — Preprocessing transforms for QSVM fraud detection.

Handles feature scaling before quantum encoding. Normalization is required
because:
  - Amplitude encoding (Eq. 1) divides by ||x||; raw features with vastly
    different scales distort the relative amplitudes.
  - Angle encoding (Eq. 2) uses Ry rotations; features must be bounded
    to produce meaningful rotation angles.

ASSUMPTION: StandardScaler is used (confidence: 0.80). The paper does not
specify the scaler. MinMaxScaler to [0, pi] is the alternative for angle
encoding. StandardScaler is used as the primary assumption.

Paper reference: Section IV-A (data preparation, implicit preprocessing).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """
    Feature scaler for QSVM fraud detection pipeline.

    Fits on training data only and applies the same transform to test data,
    preventing data leakage.

    ASSUMPTION: StandardScaler (zero mean, unit variance). Confidence: 0.80.
    TODO: Compare results with MinMaxScaler(feature_range=(0, np.pi)) which
          is more natural for angle encoding.

    Args:
        scaler_type: 'standard' (default) or 'minmax'.
        feature_range: For minmax only — target range tuple.
    """

    def __init__(
        self,
        scaler_type: str = "standard",  # ASSUMED (confidence: 0.80)
        feature_range: tuple[float, float] = (0.0, float(np.pi)),
    ) -> None:
        if scaler_type not in ("standard", "minmax"):
            raise ValueError(f"scaler_type must be 'standard' or 'minmax'; got {scaler_type!r}")

        self.scaler_type = scaler_type
        self.feature_range = feature_range
        self._scaler: Optional[StandardScaler | MinMaxScaler] = None

    def fit(self, X_train: np.ndarray) -> "FraudPreprocessor":
        """
        Fit scaler on training data.

        Args:
            X_train: [N_train, D] float64.

        Returns:
            self (fitted preprocessor).
        """
        if self.scaler_type == "standard":
            self._scaler = StandardScaler()
        else:
            self._scaler = MinMaxScaler(feature_range=self.feature_range)

        self._scaler.fit(X_train)
        logger.debug("FraudPreprocessor fitted: scaler_type=%s", self.scaler_type)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted scaler to X.

        Args:
            X: [N, D] float64.

        Returns:
            X_scaled: [N, D] float64.
        """
        if self._scaler is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return self._scaler.transform(X).astype(np.float64)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on X_train and return scaled X_train."""
        self.fit(X_train)
        return self.transform(X_train)

    def __repr__(self) -> str:
        fitted = self._scaler is not None
        return f"FraudPreprocessor(scaler_type={self.scaler_type!r}, fitted={fitted})"
