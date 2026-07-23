"""Policy gradient loss (Eq. 14) and value MSE loss (Eq. 15)."""
from __future__ import annotations

import torch


class LossFunctions:
    @staticmethod
    def policy_loss(advantages: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """Eq. 14: theta -> -(1/(tau*N)) * sum_k sum_n A(s,a) * log phi_theta(x|s).

        `advantages` and `log_probs` are expected flattened over (tau, N),
        i.e. shape [tau*N]. Advantages should be detached (no grad through
        the critic here).
        """
        assert advantages.shape == log_probs.shape, \
            f"shape mismatch: {advantages.shape} vs {log_probs.shape}"
        return -(advantages.detach() * log_probs).mean()

    @staticmethod
    def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Eq. 15: vartheta -> (1/(tau*N)) * sum (V(s) - return)^2."""
        assert values.shape == returns.shape, \
            f"shape mismatch: {values.shape} vs {returns.shape}"
        return (values - returns).pow(2).mean()
