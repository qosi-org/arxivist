"""
training/losses.py — Loss functions for GAN asset pricing.

Implements the empirical moment condition loss from Eq. (3) in Section III.A:

    L(omega | g, I_t, I_{t,i}) = (1/N) sum_i (T_i/T) ||
        (1/T_i) sum_{t in T_i} M_{t+1} R^e_{t+1,i} g_hat(I_t, I_{t,i})
    ||^2

This is a minimax loss — the SDF network minimizes it, the conditional network
maximizes it. The loss is the squared mean pricing error for each stock, weighted
by the panel weight sqrt(T_i/T), for all conditioning instruments g.

Key insight (Section III.A): The no-arbitrage condition implies this moment
should be zero for any g. By maximizing over g, we find the most violated moments.

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Eq. (3).
"""

import torch
import torch.nn as nn
from typing import Optional


class MomentConditionLoss(nn.Module):
    """
    Pricing error loss based on the no-arbitrage moment condition.

    Computes the weighted squared mean pricing error across all stocks and
    all conditioning instruments g.

    The moment condition E[M_{t+1} R^e_{t+1,i} g(I_t, I_{t,i})] = 0
    must hold for any valid pricing function g (Section II.B).

    Args:
        reduction: 'mean' (average over stocks) or 'sum'
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in ("mean", "sum"), f"Unknown reduction: {reduction}"
        self.reduction = reduction

    def forward(
        self,
        M_t: torch.Tensor,
        returns: torch.Tensor,
        g: torch.Tensor,
        panel_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pricing error loss (Eq. 3).

        L = (1/N) sum_i w_i * ||mean_t [M_{t+1,i} * R^e_{t+1,i} * g_{t,i}]||^2

        where w_i = T_i / T (fraction of periods stock i is observed).

        Args:
            M_t: [T] SDF values M_{t+1} = 1 - F_{t+1}
            returns: [T, N] excess returns R^e_{t+1,i}
            g: [T, N, num_moments] conditioning instruments
            panel_weights: [N] weights T_i/T per stock (None → equal weights)

        Returns:
            loss: scalar pricing error loss
        """
        T, N = returns.shape
        num_moments = g.shape[-1]

        # Pricing error per stock-time: M_{t+1} * R^e_{t+1,i}  →  [T, N]
        pricing_error = M_t.unsqueeze(-1) * returns  # [T, N]

        # Instrument-weighted pricing error: e_{t,i} * g_{t,i,k}  →  [T, N, K]
        # Eq. (3): M_{t+1} R^e_{t+1,i} g_hat(I_t, I_{t,i})
        weighted_error = pricing_error.unsqueeze(-1) * g  # [T, N, K]

        # Time-average per stock per moment: [N, K]
        # Eq. (3): (1/T_i) sum_{t in T_i} ...
        mean_error = weighted_error.mean(dim=0)  # [N, K]

        # Squared norm over moments: [N]
        squared_norm = (mean_error ** 2).sum(dim=-1)  # [N]

        # Apply panel weights T_i/T (Eq. 3 weighting)
        if panel_weights is not None:
            assert panel_weights.shape == (N,), f"panel_weights shape mismatch: {panel_weights.shape}"
            squared_norm = panel_weights * squared_norm

        # Average over stocks (1/N)
        if self.reduction == "mean":
            loss = squared_norm.mean()
        else:
            loss = squared_norm.sum()

        return loss

    def __repr__(self) -> str:
        return f"MomentConditionLoss(reduction={self.reduction})"


class UnconditionalMomentLoss(nn.Module):
    """
    Unconditional moment condition loss (g = constant = 1).

    Used in Step 1 of training: initialize SDF network by minimizing
    the unconditional pricing error before the adversarial game.

    This corresponds to minimizing:
        (1/N) sum_i w_i * (mean_t [M_{t+1} R^e_{t+1,i}])^2
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        M_t: torch.Tensor,
        returns: torch.Tensor,
        panel_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            M_t: [T] SDF values
            returns: [T, N] excess returns
            panel_weights: [N] optional panel weights

        Returns:
            loss: scalar unconditional pricing error
        """
        T, N = returns.shape

        # Unconditional pricing error per stock: mean_t[M * R^e] → [N]
        pricing_error = (M_t.unsqueeze(-1) * returns).mean(dim=0)  # [N]
        squared_error = pricing_error ** 2  # [N]

        if panel_weights is not None:
            squared_error = panel_weights * squared_error

        return squared_error.mean()

    def __repr__(self) -> str:
        return "UnconditionalMomentLoss()"


class LoadingRegressionLoss(nn.Module):
    """
    MSE loss for training the loading (beta) network.

    The loading network is trained to predict R^e_{t+1,i} * F_{t+1},
    approximating E_t[R^e_{t+1,i} * F_{t+1}] ∝ beta_{t,i}.

    From Section III.F: beta_{t,i} is obtained by fitting a feedforward
    network to predict R^e_{t+1} * F_{t+1}.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        beta_pred: torch.Tensor,
        returns: torch.Tensor,
        F_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            beta_pred: [T, N] predicted loadings from LoadingNetwork
            returns: [T, N] excess returns R^e_{t+1,i}
            F_t: [T] SDF factor returns F_{t+1}

        Returns:
            loss: MSE(beta_pred, R^e * F_{t+1})
        """
        # Target: R^e_{t+1,i} * F_{t+1}  [T, N]
        target = returns * F_t.unsqueeze(-1)
        return self.mse(beta_pred, target)

    def __repr__(self) -> str:
        return "LoadingRegressionLoss()"
