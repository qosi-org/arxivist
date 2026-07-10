"""
Manual implementation of Batch Normalization Transform.
Paper: Ioffe & Szegedy (2015), Algorithm 1, Section 3.
arXiv:1502.03167
"""
import torch
import torch.nn as nn


class BatchNorm1dManual(nn.Module):
    """
    Manual implementation of Batch Normalizing Transform (Algorithm 1).
    Matches paper exactly: normalize per dimension, then scale+shift.

    Training: uses mini-batch mean and variance.
    Inference: uses running population mean and variance.

    Args:
        num_features: D — number of features per activation
        epsilon: numerical stability constant (ASSUMED: 1e-5, not stated in paper)
        momentum: moving average decay for inference stats (ASSUMED: 0.9, not stated)
    """
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.9):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon  # ASSUMED: 1e-5 (confidence: 0.7)
        self.momentum = momentum  # ASSUMED: 0.9 (confidence: 0.65)

        # Learnable parameters gamma (scale) and beta (shift) — Section 3
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running stats for inference — Section 3.1
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] input tensor
        Returns:
            y: [B, D] batch-normalized output
        """
        assert x.dim() == 2, f"Expected [B, D], got {x.shape}"

        if self.training:
            # Algorithm 1: mini-batch mean and variance
            mu_B = x.mean(dim=0)           # [D] — mini-batch mean
            sigma2_B = x.var(dim=0, unbiased=False)  # [D] — mini-batch variance

            # Update running stats for inference (Section 3.1)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu_B
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sigma2_B

            # Normalize — Algorithm 1, step 3
            x_hat = (x - mu_B) / torch.sqrt(sigma2_B + self.epsilon)
        else:
            # Inference: use population statistics — Section 3.1
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

        # Scale and shift — Algorithm 1, step 4: y = gamma * x_hat + beta
        y = self.gamma * x_hat + self.beta
        return y

    def __repr__(self):
        return f"BatchNorm1dManual(num_features={self.num_features}, eps={self.epsilon})"
