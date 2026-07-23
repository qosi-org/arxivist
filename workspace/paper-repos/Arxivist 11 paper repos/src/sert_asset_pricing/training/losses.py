"""
Training loss: MSE + L1 regularization (Section 4.1.5).

"The estimation of Transformer models in this study minimises the MSE function with
an L1 regularization term (to prevent latent overfitting) through the method of
Stochastic Gradient Descent (SGD) with the Adaptive Moment Estimation (Adam) optimizer."

The L1 coefficient (lambda) is not numerically specified in the paper — this is an
ASSUMED value (SIR ambiguities[1], confidence 0.35), exposed via config `training.l1_lambda`.
"""
from __future__ import annotations

import torch
from torch import nn


class MSEWithL1(nn.Module):
    """Mean-squared-error loss plus L1 penalty over all model parameters (Section 4.1.5).

    L = (1/N) * sum((y_i - y_hat_i)^2) + lambda * sum(|theta|)

    Args:
        l1_lambda: L1 regularization coefficient. ASSUMED default 1e-5 (paper does not
            state a value; see config.yaml comment and SIR ambiguities[1]).
    """

    def __init__(self, l1_lambda: float = 1e-5) -> None:
        super().__init__()
        assert l1_lambda >= 0, f"l1_lambda must be non-negative, got {l1_lambda}"
        self.l1_lambda = l1_lambda
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Compute MSE(pred, target) + l1_lambda * sum(|theta| for theta in model.parameters()).

        Args:
            pred: predicted returns, any shape matching `target`.
            target: ground-truth returns, same shape as `pred`.
            model: the model whose parameters are L1-penalized.

        Returns:
            Scalar loss tensor.
        """
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        mse_loss = self.mse(pred, target)
        l1_penalty = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
        return mse_loss + self.l1_lambda * l1_penalty

    def __repr__(self) -> str:
        return f"MSEWithL1(l1_lambda={self.l1_lambda})"
