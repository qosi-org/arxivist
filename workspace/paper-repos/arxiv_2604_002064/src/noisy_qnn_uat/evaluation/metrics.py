"""Error metrics used throughout the paper's numerical section.

Implements: Architecture Plan module `evaluation/metrics.py`.
Paper sections: Section 4.2-4.5 (RMSE, MAE, max error), Figures S2.2/S2.6/B.5/B.6
(MAE/theoretical-bound ratio, used to empirically validate the theorems).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorMetrics:
    """Standard error metrics for comparing QNN predictions against ground truth."""

    def rmse(self, pred: np.ndarray, true: np.ndarray) -> float:
        """Root-mean-squared error."""
        assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    def mae(self, pred: np.ndarray, true: np.ndarray) -> float:
        """Mean absolute error."""
        assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"
        return float(np.mean(np.abs(pred - true)))

    def max_error(self, pred: np.ndarray, true: np.ndarray) -> float:
        """Maximum absolute error."""
        assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"
        return float(np.max(np.abs(pred - true)))

    def relative_error(self, pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Pointwise relative error, in percent. `eps` avoids division by zero."""
        assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"
        return 100.0 * np.abs(pred - true) / (np.abs(true) + eps)

    def bound_ratio(self, empirical_mae: float, theoretical_bound: float) -> float:
        """Ratio of empirical MAE to the theoretical error bound.

        A ratio < 1 confirms that the theorem's bound holds empirically at the
        tested (finite) n / noise level, as validated throughout Section 4
        (e.g. Fig. S2.2(a), S2.6, B.5, B.6: "the ratio MAE/epsilon_n lies below 1
        throughout").

        Args:
            empirical_mae: measured mean absolute error.
            theoretical_bound: the analytical bound to compare against
                (e.g. from `evaluation/hardware_bounds.py`).

        Returns:
            empirical_mae / theoretical_bound.
        """
        assert theoretical_bound > 0, "theoretical_bound must be positive"
        return empirical_mae / theoretical_bound

    def within_bound_fraction(
        self, pred: np.ndarray, true: np.ndarray, bound: float
    ) -> float:
        """Fraction of test points whose absolute error falls within `bound`.

        Reproduces the paper's "within bounds: N/N (100%)" validation statements
        (e.g. Fig. B.4, Section 4.5: "empirical MAE lies inside the analytical
        envelope on every test point").
        """
        assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"
        within = np.abs(pred - true) <= bound
        return float(np.mean(within))
