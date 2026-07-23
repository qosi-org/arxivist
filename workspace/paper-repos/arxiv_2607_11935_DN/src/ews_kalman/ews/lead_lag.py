"""
Lead-lag cross-correlation analysis and simulation lead-time detection.

Implements Section 2.4 and 3.2 of arXiv:2607.11935: optimal-lag
cross-correlation between beta (and its derivatives) and classical EWS
signals (Table 2, Figure 2); Pearson-correlation-based orthogonality testing
(Table 1); and the |z|>2 first-significant-deviation lead-time analysis used
for the simulated tipping-point validation (Table 3, Figure 3).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats


class LeadLagAnalyzer:
    """Cross-correlation lead-lag analysis between two aligned signals."""

    def __repr__(self) -> str:  # noqa: D105
        return "LeadLagAnalyzer()"

    def optimal_lag(
        self, signal_a: np.ndarray, signal_b: np.ndarray, max_lag: int = 24
    ) -> Tuple[int, float]:
        """Find the lag (in samples) at which signal_a's cross-correlation
        with signal_b is maximised in absolute value (Section 2.4, Figure 2).

        Convention: positive lag means signal_a leads signal_b (i.e.
        signal_a(t) best predicts/aligns with signal_b(t+lag)).

        Args:
            signal_a: first signal (e.g. beta), shape [N].
            signal_b: second signal (e.g. AR(1)), shape [M] (need not equal N;
                signals are truncated to the overlapping length after
                alignment at lag 0).
            max_lag: maximum absolute lag to search, in samples.

        Returns:
            (optimal_lag, correlation_at_optimal_lag).
        """
        n = min(len(signal_a), len(signal_b))
        a = (signal_a[:n] - np.mean(signal_a[:n])) / (np.std(signal_a[:n]) + 1e-12)
        b = (signal_b[:n] - np.mean(signal_b[:n])) / (np.std(signal_b[:n]) + 1e-12)

        best_lag = 0
        best_corr = 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a_seg, b_seg = a[: n - lag], b[lag:n]
            else:
                a_seg, b_seg = a[-lag:n], b[: n + lag]
            if len(a_seg) < 3:
                continue
            corr = np.corrcoef(a_seg, b_seg)[0, 1]
            if np.isnan(corr):
                continue
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_lag, float(best_corr)

    def pearson_orthogonality(
        self, beta: np.ndarray, other_ews: np.ndarray
    ) -> Tuple[float, float]:
        """Pearson correlation and p-value between two aligned signals
        (Table 1's r(beta, AR1) and r(beta, MI) columns).

        Args:
            beta: first signal, shape [N].
            other_ews: second signal, shape [M] (truncated to overlap with beta).

        Returns:
            (pearson_r, p_value).
        """
        n = min(len(beta), len(other_ews))
        r, p = stats.pearsonr(beta[:n], other_ews[:n])
        return float(r), float(p)

    def first_significant_deviation(
        self,
        signal: np.ndarray,
        baseline_mean: float,
        baseline_std: float,
        z_threshold: float = 2.0,
    ) -> Optional[int]:
        """Index of the first timestep where |z-score| exceeds `z_threshold`
        relative to a baseline mean/std (Section 2.4).

        Args:
            signal: signal to scan, shape [N].
            baseline_mean: mean of a pre-transition baseline window.
            baseline_std: std of the same baseline window.
            z_threshold: significance threshold (paper: 2.0).

        Returns:
            Index of first significant deviation, or None if the threshold
            is never crossed.
        """
        if baseline_std < 1e-12:
            return None
        z = (signal - baseline_mean) / baseline_std
        idx = np.where(np.abs(z) > z_threshold)[0]
        return int(idx[0]) if len(idx) > 0 else None

    def simulation_lead_time(
        self,
        ews_signal: np.ndarray,
        tipping_index: int,
        baseline_window: Tuple[int, int],
        z_threshold: float = 2.0,
    ) -> Optional[int]:
        """Lead time (in timesteps) between an EWS's first significant
        deviation and the known tipping point (Section 2.4, Table 3).

        lead_time = tipping_index - first_significant_deviation_index
        (positive => the signal gave advance warning before the tipping point)

        Args:
            ews_signal: signal to scan, shape [N].
            tipping_index: known tipping-point index for this simulation.
            baseline_window: (start, end) indices defining the pre-transition
                baseline used to compute mean/std for z-scoring.
            z_threshold: significance threshold (paper: 2.0).

        Returns:
            Lead time in timesteps, or None if the signal never crosses
            threshold before the tipping point.
        """
        b_start, b_end = baseline_window
        baseline = ews_signal[b_start:b_end]
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline))

        detection_idx = self.first_significant_deviation(
            ews_signal, baseline_mean, baseline_std, z_threshold
        )
        if detection_idx is None or detection_idx > tipping_index:
            return None
        return tipping_index - detection_idx
