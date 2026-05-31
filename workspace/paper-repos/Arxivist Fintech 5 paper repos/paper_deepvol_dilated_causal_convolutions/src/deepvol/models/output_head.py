"""
Output head: maps attended context to scalar volatility forecast.
Implements the output projection from Eq. 27 (Section 4.2).
# TODO: exact MLP structure not fully described — two-layer MLP assumed; confidence=0.65
"""
import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """
    Two-layer MLP that maps the attended context vector to a scalar
    day-ahead realised variance forecast (sigma^2_{T+1}).

    WARNING: low-confidence implementation — output head architecture
    inferred from Eq. 27 and Table 1 (end_channels=64). May need adjustment.

    Args:
        in_features: Attended context size (= skip_channels = 128)
        hidden_features: Hidden layer size (= end_channels = 64 from Table 1)
    """
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),          # sigma_ReLU from Eq. 27
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            sigma2_hat: [B, 1]
        """
        assert x.dim() == 2, f"Expected [B, C], got {x.shape}"
        return self.net(x)

    def __repr__(self):
        return f"OutputHead(in={self.net[0].in_features}, hidden={self.net[0].out_features})"
