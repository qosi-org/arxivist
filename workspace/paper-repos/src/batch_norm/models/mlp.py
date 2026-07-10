"""
Batch-Normalized MLP for MNIST experiment.
Paper: Section 4.1 — 3 hidden layers of 100 units, sigmoid activation, BN before nonlinearity.
arXiv:1502.03167
"""
import torch
import torch.nn as nn
from .batch_norm import BatchNorm1dManual


class BatchNormMLP(nn.Module):
    """
    3-layer fully connected network with Batch Normalization before each sigmoid.
    Implements the MNIST experiment from Section 4.1.

    Architecture per paper:
      Input (784) -> [Linear -> BN -> Sigmoid] x3 -> Linear -> Softmax
      BN applied before nonlinearity: z = g(BN(Wu)) — Section 3.2
    """
    def __init__(self, input_dim: int = 784, hidden_units: int = 100,
                 num_classes: int = 10, epsilon: float = 1e-5, momentum: float = 0.9):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_units, bias=False)  # bias subsumed by beta
        self.bn1 = BatchNorm1dManual(hidden_units, epsilon=epsilon, momentum=momentum)

        self.layer2 = nn.Linear(hidden_units, hidden_units, bias=False)
        self.bn2 = BatchNorm1dManual(hidden_units, epsilon=epsilon, momentum=momentum)

        self.layer3 = nn.Linear(hidden_units, hidden_units, bias=False)
        self.bn3 = BatchNorm1dManual(hidden_units, epsilon=epsilon, momentum=momentum)

        self.output = nn.Linear(hidden_units, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 784] flattened MNIST image
        Returns:
            logits: [B, 10]
        """
        assert x.dim() == 2, f"Expected [B, 784], got {x.shape}"

        # z = g(BN(Wu)) — Section 3.2
        x = self.sigmoid(self.bn1(self.layer1(x)))
        x = self.sigmoid(self.bn2(self.layer2(x)))
        x = self.sigmoid(self.bn3(self.layer3(x)))
        return self.output(x)

    def __repr__(self):
        return f"BatchNormMLP(hidden=100x3, activation=sigmoid, bn=before_nonlinearity)"
