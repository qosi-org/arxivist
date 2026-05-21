"""
Loss functions for SPG-UVM actor-critic training.

Implements:
  - PPOLoss:             Clipped surrogate objective (Section 3.1 / 4.1.3)
  - CriticLoss:          MSE value regression (Section 4.1.3)
  - CorrelationPenalty:  Huber penalty for pairwise correlation bounds (Section 4.1.1)

All equations from arXiv:2605.06670 unless otherwise noted.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PPOLoss(nn.Module):
    """
    PPO clipped surrogate actor loss.

    Implements Eq. (20) from arXiv:2605.06670:

        L_PPO(theta) = -E[min(r_theta * Adv, clip(r_theta, 1-eps, 1+eps) * Adv)]

    The negation converts the maximization objective to a minimization loss.
    Advantage normalization (zero mean, unit std) is applied before the loss
    (Section 4.1.3: "standardize advantages").

    Args:
        epsilon: PPO clipping parameter (default 0.2). Section 4.1.3.
    """

    def __init__(self, epsilon: float = 0.2) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, ratio: Tensor, advantage: Tensor) -> Tensor:
        """
        Args:
            ratio:     Likelihood ratio p_theta / p_theta_old, shape [B].
            advantage: Normalized advantage estimates, shape [B].

        Returns:
            Scalar PPO loss (to be minimized).
        """
        assert ratio.shape == advantage.shape, (
            f"ratio {ratio.shape} and advantage {advantage.shape} must match"
        )
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        # Negate because we want to maximize the surrogate objective
        loss = -torch.min(surr1, surr2).mean()
        return loss

    def __repr__(self) -> str:
        return f"PPOLoss(epsilon={self.epsilon})"


class CriticLoss(nn.Module):
    """
    Critic (value network) MSE regression loss.

    Implements:
        L_C(phi) = E[(V_phi(X_n) - target)^2]

    where target = e^{-r*dt} * V_{phi*}_{n+1}(F(X_n, a_n, xi_n))
    is the bootstrapped Monte Carlo target from the already-trained next step.

    Section 4.1.3 of arXiv:2605.06670.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, v_pred: Tensor, v_target: Tensor) -> Tensor:
        """
        Args:
            v_pred:   Critic predictions V_phi(X_n), shape [B] or [B, 1].
            v_target: Regression targets (discounted next values), shape [B] or [B, 1].

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(v_pred.squeeze(-1), v_target.squeeze(-1))

    def __repr__(self) -> str:
        return "CriticLoss(MSE)"


class CorrelationPenalty(nn.Module):
    """
    Huber penalty for pairwise correlation constraint violations.

    Penalizes actor outputs (at the deterministic mean action) when
    the inferred pairwise correlation rho^{ij} falls outside [rho^{ij}_min, rho^{ij}_max].

    Eq. (in Section 4.1.1) of arXiv:2605.06670:

        Psi(rho) = (beta / d(d-1)) * sum_{i<j}
            [Hub_delta((rho^{ij} - rho^{ij}_max)_+ / (rho^{ij}_max - rho^{ij}_min))
           + Hub_delta((rho^{ij}_min - rho^{ij})_+ / (rho^{ij}_max - rho^{ij}_min))]

    where Hub_delta(u) is the Huber function:
        Hub_delta(u) = u^2/2           if |u| <= delta
                       delta*(|u|-d/2) if |u| > delta

    Applied only when d >= 3 and policy_type = "continuous".
    Evaluated at the DETERMINISTIC mean action TUVM(m_theta(x)), not stochastic samples.
    (Explicitly stated in Section 4.1.1.)

    Args:
        rho_min:  Scalar lower bound for all pairwise correlations.
        rho_max:  Scalar upper bound for all pairwise correlations.
        beta:     Penalty weight (default 10). Section 4.1.1.
        delta:    Huber threshold (default 0.05). Section 4.1.1.
    """

    def __init__(
        self,
        rho_min: float,
        rho_max: float,
        beta: float = 10.0,
        delta: float = 0.05,
    ) -> None:
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_range = rho_max - rho_min
        self.beta = beta
        self.delta = delta

    def _huber(self, u: Tensor) -> Tensor:
        """
        Huber function (smooth L1):
            Hub(u) = u^2 / 2            if |u| <= delta
                     delta*(|u| - d/2)  if |u| > delta

        PyTorch's F.huber_loss uses a slightly different convention;
        we implement directly for clarity.
        """
        abs_u = u.abs()
        return torch.where(
            abs_u <= self.delta,
            0.5 * u ** 2,
            self.delta * (abs_u - 0.5 * self.delta),
        )

    def forward(self, rho: Tensor) -> Tensor:
        """
        Compute the correlation constraint penalty.

        Args:
            rho: Correlation matrices, shape [B, d, d].

        Returns:
            Scalar penalty (averaged over batch and pairs).
        """
        B, d, _ = rho.shape
        if d < 3:
            # Penalty only applied for d >= 3 (Section 4.1.1)
            return rho.new_zeros(1).squeeze()

        # Extract upper-triangular pairwise correlations (i < j)
        # upper_indices returns (row_indices, col_indices)
        idx = torch.triu_indices(d, d, offset=1, device=rho.device)
        # rho_ij: [B, d*(d-1)//2]
        rho_ij = rho[:, idx[0], idx[1]]

        n_pairs = rho_ij.shape[1]

        # Upper bound violation: max(0, rho_ij - rho_max) / range
        upper_viol = torch.relu(rho_ij - self.rho_max) / (self.rho_range + 1e-8)
        # Lower bound violation: max(0, rho_min - rho_ij) / range
        lower_viol = torch.relu(self.rho_min - rho_ij) / (self.rho_range + 1e-8)

        penalty = (
            self._huber(upper_viol) + self._huber(lower_viol)
        ).sum(dim=-1).mean()  # mean over batch

        # Normalize by d*(d-1) (number of ordered pairs = 2 * n_pairs)
        penalty = self.beta / (d * (d - 1)) * penalty * 2  # factor 2: sum over i<j counted once

        return penalty

    def __repr__(self) -> str:
        return (
            f"CorrelationPenalty(rho_min={self.rho_min}, rho_max={self.rho_max}, "
            f"beta={self.beta}, delta={self.delta})"
        )
