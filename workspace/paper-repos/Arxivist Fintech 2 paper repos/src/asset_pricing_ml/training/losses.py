"""
training/losses.py — Loss functions for Gu, Kelly, Xiu (2020).

Implements three objective functions from Section 1.2:
  - L2 (standard least squares)
  - Weighted L2 (value-weighted or time-equal-weighted)
  - Huber robust loss (for heavy-tailed financial returns)

Paper reference: Equations (4), (5), (6) in Section 1.2
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    import types
    nn = types.SimpleNamespace(Module=object)


class L2Loss(nn.Module):
    """Standard pooled OLS objective function.

    Paper Equation (4):
        L(theta) = (1/NT) * sum_it (r_it+1 - g(z_it; theta))^2

    Paper reference: Section 1.2, Equation (4)
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [N] predicted returns
            target: [N] realized excess returns
        Returns:
            Scalar mean squared error loss.
        """
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
        # Eq. (4): (1/NT) * sum (r - r_hat)^2
        return torch.mean((target - pred) ** 2)


class WeightedL2Loss(nn.Module):
    """Weighted least squares objective.

    Paper Equation (5):
        L_W(theta) = (1/NT) * sum_it w_it * (r_it+1 - g(z_it; theta))^2

    Two weighting schemes from the paper:
      - 'time_equal': w_it = 1/N_t (equal weight per month)
      - 'value':      w_it proportional to market cap of stock i at time t

    Paper reference: Section 1.2, Equation (5)
    """

    def __init__(self, weighting: str = "time_equal"):
        """
        Args:
            weighting: 'time_equal' or 'value'.
        """
        super().__init__()
        if weighting not in ("time_equal", "value"):
            raise ValueError(f"weighting must be 'time_equal' or 'value', got '{weighting}'")
        self.weighting = weighting

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:    [N] predicted returns
            target:  [N] realized excess returns
            weights: [N] observation weights (will be normalized to sum=1)
        Returns:
            Scalar weighted MSE loss.
        """
        assert pred.shape == target.shape == weights.shape
        # Normalize weights to sum to 1
        w = weights / (weights.sum() + 1e-8)
        # Eq. (5): sum_it w_it * (r - r_hat)^2
        return torch.sum(w * (target - pred) ** 2)


class HuberLoss(nn.Module):
    """Huber robust loss for heavy-tailed financial returns.

    Paper Equation (6):
        H(x; xi) = x^2          if |x| <= xi
                   2*xi*|x|-xi^2 if |x| >  xi

        L_H(theta) = (1/NT) * sum_it H(r_it+1 - g(z_it; theta), xi)

    The Huber loss is quadratic for small errors and linear for large errors,
    reducing the influence of outliers relative to standard OLS.

    Paper reference: Section 1.2, Equations (6) and (7)

    Args:
        xi: Transition point between quadratic and linear regime.
            Tuned via validation sample. Default 1.0.
    """

    def __init__(self, xi: float = 1.0):
        super().__init__()
        self.xi = xi

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [N] predicted returns
            target: [N] realized excess returns
        Returns:
            Scalar Huber loss.
        """
        assert pred.shape == target.shape
        residual = target - pred
        abs_res = residual.abs()
        xi = self.xi

        # Eq. (7): H(x; xi) = x^2 if |x|<=xi, else 2*xi*|x| - xi^2
        quadratic = residual ** 2
        linear = 2.0 * xi * abs_res - xi ** 2
        loss = torch.where(abs_res <= xi, quadratic, linear)
        return loss.mean()

    def tune_xi(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        xi_grid: list,
    ) -> float:
        """Find best xi on a validation set by minimizing Huber loss.

        Args:
            pred:    [N] predicted returns (numpy)
            target:  [N] realized returns (numpy)
            xi_grid: List of candidate xi values.

        Returns:
            Best xi value.
        """
        best_xi = xi_grid[0]
        best_loss = float("inf")
        for xi in xi_grid:
            residual = target - pred
            abs_res = np.abs(residual)
            loss = np.where(abs_res <= xi, residual**2, 2*xi*abs_res - xi**2).mean()
            if loss < best_loss:
                best_loss = loss
                best_xi = xi
        return best_xi

    def __repr__(self) -> str:
        return f"HuberLoss(xi={self.xi})"
