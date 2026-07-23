"""Critic (value function) network V_vartheta(s).

Paper reference: Section 4.4 (Eq. 5, 13, 15), Appendix B.1 / Table 6.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """2-hidden-layer tanh MLP estimating V^{pi_theta}(s) (Eq. 5).

    ASSUMED (SIR ambiguities[1], confidence 0.65): uses the same orthogonal
    initialization scheme (gain=0.01) as the policy network's non-final
    layers, since the paper only explicitly restates the policy network's
    init scheme in detail.
    """

    def __init__(self, state_dim: int, hidden_units: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.orthogonal_(layer.weight, gain=0.01)  # ASSUMED, see docstring
            nn.init.constant_(layer.bias, 0.0)
        self.act = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return V(s): [B, 1]."""
        assert state.dim() == 2, f"Expected [B, state_dim], got {tuple(state.shape)}"
        h = self.act(self.fc1(state))
        h = self.act(self.fc2(h))
        v = self.fc3(h)
        return v

    def __repr__(self) -> str:
        return "ValueNetwork(hidden=128x2, tanh)"
