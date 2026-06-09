"""
models/q_network.py — Deep Duelling Double Q-Network.

Implements the neural network architecture described in Section 5:
  - 3 feed-forward layers (shared trunk)
  - 1 LSTM layer (shared trunk)
  - Value stream V(s) and Advantage stream A(s,a) (duelling architecture)
  - Q(s,a) = V(s) + A(s,a) - mean(A(s,.))  [Eq. duelling, Wang et al. 2016]

Instantiated twice: once as main_network, once as target_network.

Paper: arXiv:2301.08688 — Section 3.2 and Section 5.
Reference: Wang et al. (2016) Duelling Network Architectures [27]
           Van Hasselt et al. (2016) Deep Double Q-learning [26]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class DuellingQNetwork(nn.Module):
    """Deep Duelling Double Q-Network for LOB trading.

    Architecture (Section 5):
        Input: [B, history_len, state_dim] (sequence of LOB observations)
        → 3 FF (Linear+ReLU) layers applied to each time step
        → 1 LSTM layer aggregating temporal history
        → Split into Value stream V(s) and Advantage stream A(s,a)
        → Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))  [Wang et al. 2016 Eq.]

    Args:
        state_dim: Dimensionality of each LOB snapshot (default 10).
        history_len: Number of past LOB states in observation (default 100).
        hidden_dim: Hidden units in FF and LSTM layers.
            ASSUMED: 256 — not stated in paper (IA-01).
        num_actions: Number of discrete actions. Paper uses 7 (Section 4.2).
        num_ff_layers: Number of feed-forward layers before LSTM. Paper states 3.
    """

    def __init__(
        self,
        state_dim: int = 10,
        history_len: int = 100,
        hidden_dim: int = 256,
        num_actions: int = 7,
        num_ff_layers: int = 3,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # ── Shared trunk: FF layers (applied per time-step) ─────────────────
        # Section 5: "3 feed-forward layers"
        ff_layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_ff_layers):
            ff_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.ff_trunk = nn.Sequential(*ff_layers)

        # ── Shared trunk: LSTM ───────────────────────────────────────────────
        # Section 5: "followed by an LSTM layer"
        # Allows memory-based policy over 100 LOB states.
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # ── Duelling streams ────────────────────────────────────────────────
        # Section 3.2, [27]: separate value and advantage branches
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, num_actions)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: State observation tensor of shape [B, history_len, state_dim].
            hidden: Optional LSTM hidden state (h, c). None → zero-initialised.

        Returns:
            q_values: [B, num_actions] Q-value estimates.
            hidden: Updated LSTM hidden state (h, c).
        """
        assert x.dim() == 3, (
            f"Expected input [B, history_len, state_dim], got {x.shape}"
        )
        assert x.size(1) == self.history_len, (
            f"history_len mismatch: expected {self.history_len}, got {x.size(1)}"
        )
        assert x.size(2) == self.state_dim, (
            f"state_dim mismatch: expected {self.state_dim}, got {x.size(2)}"
        )

        B, T, D = x.shape

        # Apply FF trunk to each time-step independently: [B*T, D] → [B*T, H]
        x_flat = x.reshape(B * T, D)
        x_ff = self.ff_trunk(x_flat)          # [B*T, hidden_dim]
        x_seq = x_ff.reshape(B, T, self.hidden_dim)  # [B, T, hidden_dim]

        # LSTM over the history sequence: output last hidden state
        lstm_out, hidden = self.lstm(x_seq, hidden)  # lstm_out: [B, T, hidden_dim]
        features = lstm_out[:, -1, :]                 # [B, hidden_dim] — last step

        # ── Duelling composition (Eq. duelling — Wang et al. 2016) ──────────
        # Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        value = self.value_stream(features)            # [B, 1]
        advantage = self.advantage_stream(features)    # [B, num_actions]
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)  # [B, num_actions]

        return q_values, hidden

    def __repr__(self) -> str:
        return (
            f"DuellingQNetwork("
            f"state_dim={self.state_dim}, "
            f"history_len={self.history_len}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_actions={self.num_actions})"
        )
