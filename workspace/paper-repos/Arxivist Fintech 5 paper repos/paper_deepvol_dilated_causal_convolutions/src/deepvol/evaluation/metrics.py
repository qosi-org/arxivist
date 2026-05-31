"""
Evaluation metrics for volatility forecasting.
All metrics from Section 3.2 of Moreno-Pino & Zohren 2024.
"""
import numpy as np
from typing import Dict


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE — Section 3.2."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE — Section 3.2."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE — Section 3.2. Scale-independent, bounded [0, 2]."""
    eps = 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + eps)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE — Section 3.2. Noise-robust loss for volatility proxies."""
    eps = 1e-8
    return float(np.mean(np.log(y_pred + eps) + y_true / (y_pred + eps)))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ME (Maximum Error) — Section 3.2."""
    return float(np.max(np.abs(y_true - y_pred)))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MedAE — outlier-robust metric, Section 3.2."""
    return float(np.median(np.abs(y_true - y_pred)))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all 6 metrics from Section 3.2."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "qlike": qlike(y_true, y_pred),
        "me": max_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
    }
