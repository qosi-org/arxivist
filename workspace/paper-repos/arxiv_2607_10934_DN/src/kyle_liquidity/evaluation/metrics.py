"""Numerical comparison helpers between simulated quantities and paper closed forms."""
from __future__ import annotations

import numpy as np


class ClosedFormComparator:
    """Compares a simulated/computed array against a paper closed-form target array."""

    def compare(self, simulated: np.ndarray, closed_form: np.ndarray) -> dict:
        simulated = np.asarray(simulated, dtype=float)
        closed_form = np.asarray(closed_form, dtype=float)
        abs_error = np.abs(simulated - closed_form)
        denom = np.abs(closed_form)
        denom = np.where(denom < 1e-12, 1.0, denom)
        rel_error = abs_error / denom
        return {
            "abs_error_mean": float(np.mean(abs_error)),
            "abs_error_max": float(np.max(abs_error)),
            "rel_error_mean": float(np.mean(rel_error)),
            "rel_error_max": float(np.max(rel_error)),
        }
