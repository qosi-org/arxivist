"""
forecast_risk.metrics.edge_ratio
=================================
Edge Ratio: measures whether a model delivers unique predictive advantage
relative to the full forecasting frontier.

Implements Section 2.4 of:
  "Quantifying the Risk-Return Tradeoff in Forecasting"
  Philippe Goulet Coulombe, arXiv: 2605.09712

Equation:
  L*_t = min_{j != M} L_{j,t}                       (frontier loss)
  e_{M,t} = L*_t - L_{M,t}                          (edge at time t)
  EdgeRatio(M) = (sum e+_{M,t} / sum e-_{M,t}) * (M-1)

The (M-1) factor normalizes for pool size: under the null of equal
informativeness, EdgeRatio ≈ 1.

Decision-theoretic interpretation: EdgeRatio = avoided regret / incurred regret.
A model never reaching the frontier receives EdgeRatio = 0 (sum e+ = 0).
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class EdgeRatioCalculator:
    """
    Computes the Edge Ratio for each model in a pool.

    Paper: Section 2.4 — Edge Ratio: Eliciting Unique Predictive Advantage
    "Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

    Args:
        eps: Guard against division by zero when sum(e-) is near zero.
    """

    def __init__(self, eps: float = 1e-10):
        self.eps = eps

    def __repr__(self) -> str:
        return f"EdgeRatioCalculator(eps={self.eps})"

    def frontier_loss(
        self, losses_matrix: np.ndarray, model_idx: int
    ) -> np.ndarray:
        """
        Compute L*_t = min_{j != M} L_{j,t}: best competitor at each period.

        Args:
            losses_matrix: [M, T] loss matrix for all M models over T periods.
            model_idx:     Index of the model under evaluation (excluded from frontier).

        Returns:
            Frontier loss series [T].
        """
        assert losses_matrix.ndim == 2, (
            f"Expected [M, T], got shape {losses_matrix.shape}"
        )
        M, T = losses_matrix.shape
        assert 0 <= model_idx < M, f"model_idx {model_idx} out of range [0, {M})"

        # Exclude the model under evaluation
        mask = np.ones(M, dtype=bool)
        mask[model_idx] = False
        competitor_losses = losses_matrix[mask, :]  # [M-1, T]

        if competitor_losses.shape[0] == 0:
            raise ValueError(
                f"No competitors available for model_idx={model_idx} in pool of M={M}."
            )

        # L*_t = min over competitors at each t
        return np.min(competitor_losses, axis=0)  # [T]

    def edge_series(
        self, losses_matrix: np.ndarray, model_idx: int
    ) -> np.ndarray:
        """
        Compute e_{M,t} = L*_t - L_{M,t} for each period.

        Positive = model M attains the frontier.
        Negative = model M underperforms relative to best competitor.

        Args:
            losses_matrix: [M, T] loss matrix.
            model_idx:     Index of model under evaluation.

        Returns:
            Edge series [T].
        """
        L_model = losses_matrix[model_idx, :]           # [T]
        L_star = self.frontier_loss(losses_matrix, model_idx)  # [T]
        return L_star - L_model  # [T]

    def compute(
        self, losses_matrix: np.ndarray, model_idx: int
    ) -> float:
        """
        Compute Edge Ratio for a single model.

        Paper Eq: EdgeRatio(M) = (sum e+_{M,t} / sum e-_{M,t}) * (M - 1)

        Args:
            losses_matrix: [M, T] loss matrix for all M models.
            model_idx:     Index of the model to evaluate.

        Returns:
            Edge Ratio scalar. Returns 0.0 if model never attains frontier.
            Returns inf if model always attains frontier (sum e- = 0).
        """
        M = losses_matrix.shape[0]
        e = self.edge_series(losses_matrix, model_idx)  # [T]

        # e+_{M,t} = max(e_{M,t}, 0) — edge wins
        e_plus = np.maximum(e, 0.0)
        # e-_{M,t} = max(-e_{M,t}, 0) — edge regrets
        e_minus = np.maximum(-e, 0.0)

        sum_plus = np.sum(e_plus)
        sum_minus = np.sum(e_minus)

        if sum_plus == 0.0:
            # Model never attains frontier → EdgeRatio = 0
            return 0.0

        # Scale by (M-1) to normalize for pool size
        # Under null of equal performance, EdgeRatio ≈ 1
        return float((sum_plus / max(sum_minus, self.eps)) * (M - 1))

    def compute_all(self, losses_matrix: np.ndarray) -> np.ndarray:
        """
        Compute Edge Ratio for every model in the pool.

        Args:
            losses_matrix: [M, T] loss matrix.

        Returns:
            Edge ratios [M].
        """
        M = losses_matrix.shape[0]
        return np.array([self.compute(losses_matrix, i) for i in range(M)])
