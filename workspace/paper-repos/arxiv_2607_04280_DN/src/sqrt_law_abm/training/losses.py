"""
Relative least-squares objective for fitting the square-root-law impact
curve I = c * Q^delta (Section 3.2, Eq. 3).

This is the paper's only "objective function" — repurposing the
ArXivist template's `training/losses.py` slot, even though nothing here is
optimized by gradient descent.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


class RelativeLeastSquaresFit:
    """Fits I = c * Q^delta by minimizing relative squared error (Eq. 3):

        min_{c,delta} sum_i ((y_i - c * x_i^delta) / y_i)^2
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Fit c and delta.

        Args:
            x: Normalized metaorder sizes (already log-binned, Section 3.2).
            y: Normalized impacts corresponding to x.

        Returns:
            (c, delta)

        Raises:
            ValueError: If fewer than 2 valid (x, y) points are provided.
        """
        assert x.shape == y.shape, f"x and y must match shapes, got {x.shape} vs {y.shape}"
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            raise ValueError(
                f"Need at least 2 valid (x, y) points to fit I = c*Q^delta, got {len(x)}"
            )

        def model(xx, c, delta):
            return c * np.power(xx, delta)

        def relative_residuals(params, xx, yy):
            c, delta = params
            pred = model(xx, c, delta)
            return (yy - pred) / yy

        from scipy.optimize import least_squares

        result = least_squares(
            relative_residuals, x0=[1.0, 0.5], args=(x, y), bounds=([1e-6, 0.01], [100.0, 3.0])
        )
        c, delta = result.x
        return float(c), float(delta)
