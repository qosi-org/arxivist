"""
evaluation/metrics.py
=======================
RMSE, RankIC, Directional Accuracy, Weighted R2 -- the four metrics
reported in Table 2/3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import spearmanr


class ForecastMetrics:
    """Stateless collection of the paper's Table 2/3 evaluation metrics."""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error."""
        return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

    @staticmethod
    def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Rank Information Coefficient: Spearman rank correlation."""
        corr, _ = spearmanr(y_true, y_pred)
        return float(corr) if corr is not None and not np.isnan(corr) else 0.0

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraction of predictions with the same sign as the true value."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    @staticmethod
    def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Competition-style weighted R^2 (Section 3.1: Jane Street's official
        weighted R2 evaluation metric, using the dataset's 'weight' column;
        uniform weights (all ones) used for the other 3 datasets, which do
        not provide a weight column).
        """
        y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
        if weights is None:
            weights = np.ones_like(y_true)
        weights = np.asarray(weights, dtype=np.float64)
        numerator = np.sum(weights * (y_true - y_pred) ** 2)
        denominator = np.sum(weights * y_true ** 2)
        if denominator == 0:
            return float("nan")
        return float(1.0 - numerator / denominator)
