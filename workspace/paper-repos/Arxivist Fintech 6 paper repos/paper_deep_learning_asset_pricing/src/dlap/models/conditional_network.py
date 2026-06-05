"""
models/conditional_network.py — Adversarial conditional network (g function).

Implements the Conditional Network described in Section III.D. This network acts
as the adversary in the GAN: it selects the moment conditions that are hardest
to price, forcing the SDF network to explain them.

From the GAN minimax objective (Eq. 2):
    min_omega max_g (1/N) sum_j ||E[(1 - sum_i omega R^e_{t+1,i}) R^e_{t+1,j} g(I_t, I_{t,j})]||^2

The conditional network g(I_t, I_{t,j}) identifies the assets and portfolio
strategies (moments) that are the most mispriced.

Hyperparameters from Table I (optimal):
    CHL=0 (no hidden layers — linear transform of inputs)
    CHU=8 (8 moment conditions output)
    CSMV=32 (32 macro hidden states for conditional network)

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Section III.D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalNetwork(nn.Module):
    """
    Adversarial network that computes conditioning instruments g_{t,j}.

    CHL=0 means no hidden layers — the network is essentially a linear
    transformation of [h_t^g, I_{t,j}] producing CHU=8 moment conditions.

    The 8 moments can capture:
    - Pricing errors of long-short portfolios based on characteristic information
    - Portfolio payoffs conditional on macroeconomic states

    Args:
        macro_state_dim: dimension of conditional LSTM hidden states (CSMV=32)
        firm_char_dim: number of firm characteristics (46)
        num_moments: number of moment conditions to generate (CHU=8)
        num_layers: hidden layers in conditional network (CHL=0)
        hidden_units: hidden units if num_layers > 0
        dropout: dropout rate
    """

    def __init__(
        self,
        macro_state_dim: int = 32,
        firm_char_dim: int = 46,
        num_moments: int = 8,
        num_layers: int = 0,
        hidden_units: int = 8,
        dropout: float = 0.95,
    ) -> None:
        super().__init__()
        self.macro_state_dim = macro_state_dim
        self.firm_char_dim = firm_char_dim
        self.num_moments = num_moments
        self.input_dim = macro_state_dim + firm_char_dim

        if num_layers == 0:
            # CHL=0: purely linear transform [h_t^g, I_{t,j}] → g (CHU outputs)
            self.network = nn.Linear(self.input_dim, num_moments)
        else:
            layers = []
            in_dim = self.input_dim
            for _ in range(num_layers):
                layers.extend([nn.Linear(in_dim, hidden_units), nn.ReLU()])
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                in_dim = hidden_units
            layers.append(nn.Linear(in_dim, num_moments))
            self.network = nn.Sequential(*layers)

    def forward(self, h_t_g: torch.Tensor, firm_chars: torch.Tensor) -> torch.Tensor:
        """
        Compute conditioning instruments for each stock-time pair.

        # WARNING: low-confidence implementation (conf=0.60)
        # TODO: exact normalization constraint on g(.) not specified in paper.
        # Using L2 normalization per the general normalization described in Section III.B.
        # Alternatives: tanh squashing, no normalization.

        Args:
            h_t_g: [T, macro_state_dim] conditional LSTM hidden states
            firm_chars: [T, N, firm_char_dim] firm characteristics

        Returns:
            g: [T, N, num_moments] conditioning instruments (L2-normalized)
        """
        T, N, K = firm_chars.shape
        assert K == self.firm_char_dim, f"Expected firm_char_dim={self.firm_char_dim}, got {K}"
        assert h_t_g.shape[-1] == self.macro_state_dim, (
            f"Expected macro_state_dim={self.macro_state_dim}, got {h_t_g.shape[-1]}"
        )

        h_expanded = h_t_g.unsqueeze(1).expand(T, N, self.macro_state_dim)
        x = torch.cat([h_expanded, firm_chars], dim=-1)
        x_flat = x.reshape(T * N, self.input_dim)

        g_flat = self.network(x_flat)  # [T*N, num_moments]

        # L2 normalize — ensures g is a bounded conditioning function
        # TODO: verify normalization choice against paper (conf=0.60)
        g_flat = F.normalize(g_flat, p=2, dim=-1)

        g = g_flat.reshape(T, N, self.num_moments)
        return g

    def __repr__(self) -> str:
        return (
            f"ConditionalNetwork(input_dim={self.input_dim}, "
            f"num_moments={self.num_moments})"
        )
