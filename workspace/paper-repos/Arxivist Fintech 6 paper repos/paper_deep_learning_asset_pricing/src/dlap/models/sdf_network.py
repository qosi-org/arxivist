"""
models/sdf_network.py — Feedforward networks for SDF weights and risk loadings.

Implements the Feedforward Network (FFN) described in Section III.B used as:
  1. SDFNetwork: estimates SDF portfolio weights omega_{t,i} = omega(I_t, I_{t,i})
  2. LoadingNetwork: estimates risk loadings beta_{t,i} ∝ E_t[R^e_{t+1,i} * F_{t+1}]

Architecture (Table I, optimal hyperparameters):
  - HL=2 hidden layers
  - HU=64 hidden units per layer
  - ReLU activation (Section III.B)
  - DR=0.95 dropout

Input: concatenation of [macro_hidden_states h_t, firm_characteristics I_{t,i}]
       = [4 + 46] = 50-dimensional vector per stock-time pair

From Section III.B: FFN is a universal approximator f: R^K → R.
The general functional form captures arbitrary non-linearities and interaction
effects between characteristics and macroeconomic variables.

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Section III.B.
"""

import torch
import torch.nn as nn
from typing import List, Optional


def _make_ffn_layers(
    input_dim: int,
    hidden_units: int,
    num_layers: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """
    Build a feedforward network with ReLU activations and dropout.

    Architecture from Section III.B and Figures 2 & 3:
        x^(l) = ReLU(W^(l-1)^T x^(l-1) + w_0^(l-1))
        y = W^(L)^T x^(L) + w_0^(L)
    """
    layers: List[nn.Module] = []
    in_dim = input_dim
    for l in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        in_dim = hidden_units
    layers.append(nn.Linear(in_dim, output_dim))  # output layer: linear
    return nn.Sequential(*layers)


class SDFNetwork(nn.Module):
    """
    Feedforward network that computes SDF portfolio weights omega_{t,i}.

    The SDF factor weight for stock i at time t is:
        omega_{t,i} = omega(I_t, I_{t,i})
    where I_t = h_t (LSTM hidden states from macro) and I_{t,i} are firm chars.

    The SDF itself is constructed as (Section II.A):
        M_{t+1} = 1 - sum_i omega_{t,i} * R^e_{t+1,i} = 1 - omega_t^T R^e_{t+1}

    The corresponding SDF factor (tangency portfolio) is:
        F_{t+1} = omega_t^T R^e_{t+1}

    Hyperparameters from Table I (optimal):
        HL=2, HU=64, DR=0.95

    Args:
        macro_state_dim: dimension of h_t (4 for SDF network, from SMV=4)
        firm_char_dim: number of firm characteristics (46)
        num_layers: number of hidden layers (HL=2)
        hidden_units: hidden units per layer (HU=64)
        dropout: dropout rate (DR=0.95)
    """

    def __init__(
        self,
        macro_state_dim: int = 4,
        firm_char_dim: int = 46,
        num_layers: int = 2,
        hidden_units: int = 64,
        dropout: float = 0.95,
    ) -> None:
        super().__init__()
        self.macro_state_dim = macro_state_dim
        self.firm_char_dim = firm_char_dim
        self.input_dim = macro_state_dim + firm_char_dim  # [h_t, I_{t,i}]

        self.network = _make_ffn_layers(
            input_dim=self.input_dim,
            hidden_units=hidden_units,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, h_t: torch.Tensor, firm_chars: torch.Tensor) -> torch.Tensor:
        """
        Compute SDF weights for each stock-time pair.

        Args:
            h_t: [T, macro_state_dim] macroeconomic hidden states at each t
            firm_chars: [T, N, firm_char_dim] firm characteristics

        Returns:
            omega: [T, N] SDF portfolio weights per stock per time
        """
        T, N, K = firm_chars.shape
        assert K == self.firm_char_dim, f"Expected firm_char_dim={self.firm_char_dim}, got {K}"
        assert h_t.shape == (T, self.macro_state_dim), (
            f"Expected h_t shape [{T}, {self.macro_state_dim}], got {h_t.shape}"
        )

        # Expand h_t to each stock: [T, macro_state_dim] → [T, N, macro_state_dim]
        h_expanded = h_t.unsqueeze(1).expand(T, N, self.macro_state_dim)

        # Concatenate [h_t, I_{t,i}]: [T, N, input_dim]
        x = torch.cat([h_expanded, firm_chars], dim=-1)

        # Reshape to [T*N, input_dim] for batch processing
        x_flat = x.reshape(T * N, self.input_dim)

        # Forward through FFN: [T*N, 1]
        omega_flat = self.network(x_flat)

        # Reshape to [T, N]
        omega = omega_flat.squeeze(-1).reshape(T, N)
        return omega

    def __repr__(self) -> str:
        return (
            f"SDFNetwork(input_dim={self.input_dim}, "
            f"layers={len([m for m in self.network if isinstance(m, nn.Linear)])}, "
            f"output=omega[T,N])"
        )


class LoadingNetwork(nn.Module):
    """
    Feedforward network that estimates risk loadings beta_{t,i}.

    From Section III.F: beta_{t,i} is obtained by fitting a feedforward network
    to predict R^e_{t+1,i} * F_{t+1}, estimating E_t[R^e_{t+1,i} * F_{t+1}].

    Note: this loading estimate is only proportional to the population value beta,
    but this is sufficient for projecting on the systematic and non-systematic component.
    The advantage over estimating E_t[R^e] directly is that second moments have
    better signal-to-noise than first moments (Section II.B).

    Architecture: same as SDFNetwork (Section III.F)
    """

    def __init__(
        self,
        macro_state_dim: int = 4,
        firm_char_dim: int = 46,
        num_layers: int = 2,
        hidden_units: int = 64,
        dropout: float = 0.95,
    ) -> None:
        super().__init__()
        self.macro_state_dim = macro_state_dim
        self.firm_char_dim = firm_char_dim
        self.input_dim = macro_state_dim + firm_char_dim

        self.network = _make_ffn_layers(
            input_dim=self.input_dim,
            hidden_units=hidden_units,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, h_t: torch.Tensor, firm_chars: torch.Tensor) -> torch.Tensor:
        """
        Estimate risk loadings beta_{t,i} ∝ E_t[R^e_{t+1,i} * F_{t+1}].

        Args:
            h_t: [T, macro_state_dim]
            firm_chars: [T, N, firm_char_dim]

        Returns:
            beta: [T, N] risk loading estimates
        """
        T, N, K = firm_chars.shape
        h_expanded = h_t.unsqueeze(1).expand(T, N, self.macro_state_dim)
        x = torch.cat([h_expanded, firm_chars], dim=-1).reshape(T * N, self.input_dim)
        beta_flat = self.network(x)
        return beta_flat.squeeze(-1).reshape(T, N)

    def __repr__(self) -> str:
        return f"LoadingNetwork(input_dim={self.input_dim}, output=beta[T,N])"
