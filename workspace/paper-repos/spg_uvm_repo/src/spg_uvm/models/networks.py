"""
Actor and Critic neural network architectures for SPG-UVM.

Both are shallow feedforward MLPs with:
  - One hidden layer of 32 ELU units (Section 4.1.3 of arXiv:2605.06670)
  - Layer normalization on inputs (Section 4.1.3)
  - Identity output activation

Actor output dimension depends on policy type:
  - Continuous policy: d*(d+1)//2  (volatility + correlation latent mean)
  - Bang-bang policy:  d            (Bernoulli parameters per asset)

Critic output: scalar (value function estimate).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _ShallowMLP(nn.Module):
    """
    Base shallow MLP used by both actor and critic.
    Architecture: LayerNorm → Linear(d, hidden) → ELU → Linear(hidden, out_dim)

    Section 4.1.3: "feedforward neural networks with one hidden layer, using the
    nonlinear ELU activation function on the hidden layer and identity activation
    on the output layer ... layer normalization."
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_units: int = 32) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)          # Section 4.1.3: layer normalization
        self.hidden = nn.Linear(in_dim, hidden_units)
        self.elu = nn.ELU()
        self.output = nn.Linear(hidden_units, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: State tensor, shape [B, in_dim].

        Returns:
            Output tensor, shape [B, out_dim].
        """
        x = self.layer_norm(x)      # normalize inputs
        x = self.elu(self.hidden(x))
        return self.output(x)       # identity output activation


class ActorNetwork(nn.Module):
    """
    Actor (policy) network for SPG-UVM.

    For the continuous policy: outputs mean m_theta(x) of the latent Gaussian
    in R^{d(d+1)/2}. The stochastic action is z ~ N(m_theta(x), lambda*I),
    then transformed through TUVM to (sigma, rho).

    For the bang-bang policy: outputs Bernoulli probabilities q_i(x) in (0,1)^d
    via sigmoid. Each q_i is the probability of choosing sigma_max^i.

    Section 4.1.3 / Section 3.2 of arXiv:2605.06670.

    Args:
        d:             Number of assets.
        hidden_units:  Neurons in the hidden layer (default: 32).
        policy_type:   "continuous" or "bangbang".
    """

    def __init__(self, d: int, hidden_units: int = 32, policy_type: str = "continuous") -> None:
        super().__init__()
        self.d = d
        self.policy_type = policy_type

        if policy_type == "continuous":
            # Output: mean of Gaussian in R^{d(d+1)/2}
            # = d volatility components + d(d-1)/2 correlation partial-corr components
            out_dim = d * (d + 1) // 2
        elif policy_type == "bangbang":
            # Output: d Bernoulli probabilities (one per asset)
            out_dim = d
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        self.out_dim = out_dim
        self.net = _ShallowMLP(d, out_dim, hidden_units)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Asset prices (or log-prices), shape [B, d].

        Returns:
            For continuous: latent mean m_theta(x), shape [B, d*(d+1)//2].
            For bangbang:   Bernoulli logits (raw), shape [B, d].
        """
        assert x.dim() == 2 and x.shape[1] == self.d, (
            f"Expected [B, {self.d}], got {x.shape}"
        )
        out = self.net(x)

        if self.policy_type == "bangbang":
            # Apply sigmoid to get Bernoulli probabilities in (0,1)
            # Section 3.2.2: "q^i_theta : (0,+inf)^d -> (0,1)"
            out = torch.sigmoid(out)

        return out  # [B, out_dim]

    def get_mean(self, x: Tensor) -> Tensor:
        """Return mean (for continuous) or Bernoulli params (for bang-bang)."""
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"ActorNetwork(d={self.d}, out_dim={self.out_dim}, "
            f"policy={self.policy_type})"
        )


class CriticNetwork(nn.Module):
    """
    Critic (value) network for SPG-UVM.

    Approximates V_n(x) — the value function at time step n.
    Outputs a scalar for each state x.

    Section 4.1.3 of arXiv:2605.06670.

    Args:
        d:            Number of assets.
        hidden_units: Neurons in the hidden layer (default: 32).
    """

    def __init__(self, d: int, hidden_units: int = 32) -> None:
        super().__init__()
        self.d = d
        self.net = _ShallowMLP(d, 1, hidden_units)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Asset prices, shape [B, d].

        Returns:
            Value estimates V_phi(x), shape [B, 1].
        """
        assert x.dim() == 2 and x.shape[1] == self.d, (
            f"Expected [B, {self.d}], got {x.shape}"
        )
        return self.net(x)  # [B, 1]

    def __repr__(self) -> str:
        return f"CriticNetwork(d={self.d})"
