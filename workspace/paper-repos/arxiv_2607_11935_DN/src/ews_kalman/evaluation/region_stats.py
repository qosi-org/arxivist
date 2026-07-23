"""
Region-level summary statistics reproducing Table 1 (orthogonality: |beta|,
sigma_beta, r(beta,AR1), r(beta,MI), transition count) and Table 2
(systematic lead-lag between beta/derivatives and each classical EWS signal)
of arXiv:2607.11935.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ews_kalman.ews.lead_lag import LeadLagAnalyzer


class RegionSummaryComputer:
    """Computes Table-1- and Table-2-style summary rows for one region."""

    def __init__(self) -> None:
        self._lead_lag = LeadLagAnalyzer()

    def __repr__(self) -> str:  # noqa: D105
        return "RegionSummaryComputer()"

    def count_regime_transitions(self, beta_double_prime: np.ndarray) -> int:
        """Number of beta''=0 zero-crossings (Section 3.1, "Transitions" column).

        Args:
            beta_double_prime: smoothed second derivative of beta, shape [N].

        Returns:
            Count of sign changes in beta_double_prime.
        """
        signs = np.sign(beta_double_prime)
        signs = signs[signs != 0]
        if len(signs) < 2:
            return 0
        return int(np.sum(signs[1:] != signs[:-1]))

    def compute_table1_row(
        self, beta: np.ndarray, ar1_T: np.ndarray, mi: np.ndarray, beta_double_prime: np.ndarray
    ) -> Dict[str, float]:
        """Compute one Table 1 row: N, |beta|, sigma_beta, r(beta,AR1),
        r(beta,MI), and transition count.

        Args:
            beta: smoothed elasticity time series, shape [N].
            ar1_T: rolling AR(1) of temperature, shape [M] (may differ in
                length from beta due to the rolling window; truncated to
                overlap internally).
            mi: rolling mutual information T-q, shape [M].
            beta_double_prime: smoothed second derivative of beta, shape [N].

        Returns:
            Dict with keys: N, abs_beta_mean, sigma_beta, r_beta_ar1,
            r_beta_ar1_pvalue, r_beta_mi, r_beta_mi_pvalue, n_transitions.
        """
        r_ar1, p_ar1 = self._lead_lag.pearson_orthogonality(beta, ar1_T)
        r_mi, p_mi = self._lead_lag.pearson_orthogonality(beta, mi)

        return {
            "N": len(beta),
            "abs_beta_mean": float(np.mean(np.abs(beta))),
            "sigma_beta": float(np.std(beta)),
            "r_beta_ar1": r_ar1,
            "r_beta_ar1_pvalue": p_ar1,
            "r_beta_mi": r_mi,
            "r_beta_mi_pvalue": p_mi,
            "n_transitions": self.count_regime_transitions(beta_double_prime),
        }

    def compute_table2_row(
        self, beta_derivatives: Dict[str, np.ndarray], classical_signals: Dict[str, np.ndarray], max_lag: int = 24
    ) -> Dict[str, Dict[str, float]]:
        """Compute one Table 2 row: optimal lag (months) and direction
        between each beta-derivative and each classical EWS signal.

        Args:
            beta_derivatives: dict with keys 'beta', 'beta_prime',
                'beta_double_prime', 'beta_triple_prime', each shape [N].
            classical_signals: dict with keys such as 'AR1_T', 'AR1_q',
                'Var_T', 'Var_q', 'MI', 'PermEnt_T', each shape [M].
            max_lag: maximum absolute lag to search (paper: 24 months).

        Returns:
            Nested dict: {beta_derivative_name: {signal_name: {'lag': int,
            'direction': 'lead'|'lag', 'correlation': float}}}.
        """
        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for beta_name, beta_series in beta_derivatives.items():
            results[beta_name] = {}
            for signal_name, signal_series in classical_signals.items():
                lag, corr = self._lead_lag.optimal_lag(beta_series, signal_series, max_lag=max_lag)
                direction = "lead" if lag > 0 else ("lag" if lag < 0 else "coincident")
                results[beta_name][signal_name] = {
                    "lag": lag,
                    "direction": direction,
                    "correlation": corr,
                }
        return results
