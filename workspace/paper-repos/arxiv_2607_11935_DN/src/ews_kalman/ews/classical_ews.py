"""
Classical rolling-window eigenvalue-based early-warning signals: AR(1),
variance, permutation entropy, and mutual information.

Implements Section 2.3 of arXiv:2607.11935. AR(1) is estimated via linear
regression of x_t on x_{t-1} within a rolling window; variance is the window
sample variance; permutation entropy uses the standard Bandt-Pompe
ordinal-pattern method (embedding dimension m=3, per the paper); mutual
information uses a k-nearest-neighbor (Kraskov-style) estimator (paper names
"nearest-neighbor regression" without giving k -- ASSUMED k=3, SIR
implementation_assumption confidence 0.5).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.feature_selection import mutual_info_regression


class ClassicalEWS:
    """Computes classical rolling-window early-warning signals."""

    def __repr__(self) -> str:  # noqa: D105
        return "ClassicalEWS()"

    def rolling_ar1(self, x: np.ndarray, window: int = 24) -> np.ndarray:
        """Lag-1 autocorrelation via linear regression of x_t on x_{t-1}
        within each rolling window (Section 2.3).

        Args:
            x: input series, shape [N].
            window: rolling window length in samples (paper: 24 months).

        Returns:
            Array of length N - window + 1 with the AR(1) coefficient
            estimated in each window (window w covers x[w-window+1 : w+1]).
        """
        N = len(x)
        n_out = N - window + 1
        ar1 = np.zeros(n_out)
        for i in range(n_out):
            seg = x[i : i + window]
            x_t = seg[1:]
            x_tm1 = seg[:-1]
            if np.std(x_tm1) < 1e-12:
                ar1[i] = 0.0
                continue
            # simple OLS slope: cov(x_t, x_{t-1}) / var(x_{t-1})
            cov = np.mean((x_t - x_t.mean()) * (x_tm1 - x_tm1.mean()))
            var = np.var(x_tm1)
            ar1[i] = cov / var if var > 1e-12 else 0.0
        return ar1

    def rolling_variance(self, x: np.ndarray, window: int = 24) -> np.ndarray:
        """Rolling sample variance (Section 2.3).

        Args:
            x: input series, shape [N].
            window: rolling window length (paper: 24 months).

        Returns:
            Array of length N - window + 1.
        """
        N = len(x)
        n_out = N - window + 1
        return np.array([np.var(x[i : i + window]) for i in range(n_out)])

    def _ordinal_pattern_distribution(self, seg: np.ndarray, embedding_dim: int) -> np.ndarray:
        """Bandt-Pompe ordinal-pattern probability distribution for one window."""
        m = embedding_dim
        n_patterns_possible = len(seg) - m + 1
        pattern_counts: Dict[tuple, int] = {}
        for i in range(n_patterns_possible):
            window_vals = seg[i : i + m]
            # argsort gives the ordinal pattern (rank order)
            pattern = tuple(np.argsort(window_vals))
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        counts = np.array(list(pattern_counts.values()), dtype=float)
        return counts / counts.sum()

    def rolling_permutation_entropy(
        self, x: np.ndarray, embedding_dim: int = 3, window: int = 36
    ) -> np.ndarray:
        """Bandt-Pompe ordinal-pattern permutation entropy per rolling window
        (Section 2.3, embedding dimension m=3).

        H_perm = -sum_i p_i * log(p_i), normalised by log(m!) to lie in [0, 1].

        Args:
            x: input series, shape [N].
            embedding_dim: m (paper: 3).
            window: rolling window length (paper: 36 months).

        Returns:
            Array of length N - window + 1, each value the normalised
            permutation entropy of that window.
        """
        import math

        N = len(x)
        n_out = N - window + 1
        max_entropy = np.log(math.factorial(embedding_dim))
        result = np.zeros(n_out)
        for i in range(n_out):
            seg = x[i : i + window]
            probs = self._ordinal_pattern_distribution(seg, embedding_dim)
            h = -np.sum(probs * np.log(probs))
            result[i] = h / max_entropy if max_entropy > 0 else 0.0
        return result

    def rolling_mutual_information(
        self, x: np.ndarray, y: np.ndarray, window: int = 36, n_neighbors: int = 3
    ) -> np.ndarray:
        """k-nearest-neighbor mutual information estimate per rolling window
        (Section 2.3, "nearest-neighbor regression"; k ASSUMED=3, SIR
        implementation_assumption confidence 0.5).

        Args:
            x, y: aligned input series, shape [N] each.
            window: rolling window length (paper: 36 months).
            n_neighbors: k for the kNN-based MI estimator (scikit-learn's
                `mutual_info_regression`, a Kraskov-style estimator).

        Returns:
            Array of length N - window + 1.
        """
        N = len(x)
        n_out = N - window + 1
        result = np.zeros(n_out)
        for i in range(n_out):
            x_seg = x[i : i + window].reshape(-1, 1)
            y_seg = y[i : i + window]
            if np.std(x_seg) < 1e-12 or np.std(y_seg) < 1e-12:
                result[i] = 0.0
                continue
            mi = mutual_info_regression(
                x_seg, y_seg, n_neighbors=min(n_neighbors, window - 1), random_state=0
            )
            result[i] = mi[0]
        return result
