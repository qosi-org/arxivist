"""
Evaluation metrics (Sec 4.3, Sec 5.3): MAE, RMSE, MAPE, R^2.

Also provides calibration-plot and deviation-histogram data prep matching
Fig. 7 (predicted vs. true SPR distance) and Fig. 8 (absolute and
normalized deviation histograms).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionMetrics:
    """Computes the four metrics reported in the paper (Sec 4.3)."""

    def compute(self, preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
        """
        Args:
            preds: [N] predicted SPR distances.
            targets: [N] ground-truth SPR distances.
            eps: small constant to avoid division by zero in MAPE for
                zero-distance control pairs (Sec 3: "two zero-distance
                control blocks NJ vs. NJ and UPGMA vs. UPGMA").

        Returns:
            {"mae":.., "rmse":.., "mape":.., "r2":..}
        """
        mae = mean_absolute_error(targets, preds)
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        mape = float(np.mean(np.abs((targets - preds) / (np.abs(targets) + eps))) * 100.0)
        r2 = r2_score(targets, preds) if len(np.unique(targets)) > 1 else float("nan")
        return {"mae": float(mae), "rmse": rmse, "mape": mape, "r2": float(r2)}

    def calibration_data(self, preds: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
        """Data for reproducing Fig. 7 (model-predicted vs. true SPR, y=x reference)."""
        return {"true": targets, "predicted": preds}

    def deviation_histograms(self, preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray]:
        """Data for reproducing Fig. 8: absolute deviation and normalized deviation."""
        absolute_deviation = preds - targets
        normalized_deviation = absolute_deviation / (targets + eps)
        return {"absolute_deviation": absolute_deviation, "normalized_deviation": normalized_deviation}
