"""
forecast_risk.models.base
===========================
Abstract base class for all forecasting models in the evaluation suite.

Paper: Section 3 — Application 1: Predictive Personalities & Macro Forecasting
"Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """
    Abstract interface all forecasting models must implement.

    All models produce point forecasts (scalars) for direct multi-step forecasting
    where a separate model is estimated per horizon h (Paper Section 3).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model on training data.

        Args:
            X: Predictor matrix [T_train, N_features].
            y: Target vector [T_train].
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point forecasts.

        Args:
            X: Predictor matrix [T_pred, N_features].

        Returns:
            Forecasts [T_pred].
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name used in results tables."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
