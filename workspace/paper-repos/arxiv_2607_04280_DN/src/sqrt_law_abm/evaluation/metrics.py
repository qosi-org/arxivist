"""
Impact-curve fitting (Section 3.2), Hill tail-exponent estimation (Section 4.4,
Figure 8), LOB depth-profile exponent gamma, and the three competing
theoretical predictions for delta (GGPS, FGLW, LOB walking; Section 4.4 /
5.2, Eqs. 4-5).
"""

from __future__ import annotations

import numpy as np

from sqrt_law_abm.training.losses import RelativeLeastSquaresFit


class ImpactCurveFitter:
    """Log-bins normalized (Q, I) pairs and fits I = c * Q^delta (Section 3.2)."""

    def __init__(self):
        self._fit = RelativeLeastSquaresFit()

    def log_bin_and_fit(
        self, q_norm: np.ndarray, i_norm: np.ndarray, n_bins: int = 20, min_pts: int = 30
    ) -> dict:
        """Log-bin then fit the impact curve.

        Args:
            q_norm: Normalized metaorder sizes (Q/V_D).
            i_norm: Normalized impacts (I/sigma_D).
            n_bins: Number of logarithmic bins (Sec. 3.2: 20).
            min_pts: Minimum points required per bin to keep it (Sec. 3.2: >=30).

        Returns:
            {"c": float, "delta": float, "n_meta": int, "bin_x": np.ndarray, "bin_y": np.ndarray}
        """
        assert q_norm.shape == i_norm.shape, "q_norm and i_norm must have the same shape"
        n_meta = len(q_norm)
        if n_meta < min_pts:
            raise ValueError(
                f"Only {n_meta} metaorders available, need at least {min_pts} to bin reliably"
            )

        log_q = np.log10(q_norm)
        bin_edges = np.linspace(log_q.min(), log_q.max(), n_bins + 1)
        bin_idx = np.digitize(log_q, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bin_x, bin_y = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() >= min_pts:
                bin_x.append(np.mean(q_norm[mask]))
                bin_y.append(np.mean(i_norm[mask]))

        bin_x, bin_y = np.array(bin_x), np.array(bin_y)
        if len(bin_x) < 2:
            raise ValueError(
                f"Only {len(bin_x)} bins had >= {min_pts} points; cannot fit a power law. "
                "Try more metaorders (larger n_steps / more stocks)."
            )

        c, delta = self._fit.fit(bin_x, bin_y)
        return {"c": c, "delta": delta, "n_meta": n_meta, "bin_x": bin_x, "bin_y": bin_y}


class TailExponentEstimator:
    """Hill estimator for power-law tail exponents (Section 4.4, Figure 8)."""

    def hill_estimator(self, values: np.ndarray, tail_fraction: float = 0.1) -> float:
        """Estimate the tail exponent of P(X > x) ~ x^-alpha via the Hill estimator.

        Args:
            values: Positive samples (e.g. metaorder sizes or child-order counts).
            tail_fraction: Fraction of largest order statistics to use (0.1 = top 10%).

        Returns:
            Estimated tail exponent alpha (or beta, depending on context).

        Raises:
            ValueError: If fewer than 10 values are provided.
        """
        values = np.asarray(values)
        values = values[values > 0]
        if len(values) < 10:
            raise ValueError(f"Need at least 10 positive values, got {len(values)}")

        sorted_vals = np.sort(values)[::-1]
        k = max(2, int(len(sorted_vals) * tail_fraction))
        top_k = sorted_vals[:k]
        x_k1 = sorted_vals[k]  # (k+1)-th order statistic as the threshold
        if x_k1 <= 0:
            x_k1 = sorted_vals[-1]
        log_ratios = np.log(top_k / x_k1)
        hill = 1.0 / np.mean(log_ratios[log_ratios > 0])
        return float(hill)


class TheoryPredictors:
    """The three competing quantitative predictions for delta (Section 4.4, Eqs. 4-5)."""

    def ggps_delta(self, beta: float) -> float:
        """GGPS prediction: delta = beta - 1 (metaorder-size tail exponent)."""
        return beta - 1.0

    def fglw_delta(self, alpha: float) -> float:
        """FGLW prediction: delta = alpha - 1 (child-order-count tail exponent)."""
        return alpha - 1.0

    def lob_walking_delta(self, gamma: float) -> float:
        """LOB-walking prediction: delta = 1 / (1 + gamma) (visible depth-profile exponent)."""
        return 1.0 / (1.0 + gamma)


def estimate_depth_profile_gamma(depth_profile: np.ndarray) -> float:
    """Fit V(q) ~ q^(1+gamma) to a cumulative depth profile via log-log regression.

    Args:
        depth_profile: Cumulative depth at increasing distance from best
            price (LimitOrderBook.depth_profile output).

    Returns:
        Estimated gamma.
    """
    q = np.arange(1, len(depth_profile) + 1)
    mask = depth_profile > 0
    if mask.sum() < 2:
        return 1.0  # ASSUMED fallback: cannot estimate from a near-empty book
    log_q = np.log(q[mask])
    log_v = np.log(depth_profile[mask])
    slope, _ = np.polyfit(log_q, log_v, 1)
    return float(slope - 1.0)
