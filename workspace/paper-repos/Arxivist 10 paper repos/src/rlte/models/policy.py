"""Actor (policy) networks: LogisticNormalPolicy (primary, LN) and
DirichletPolicy (benchmark, DR).

Paper reference: Section 4.2-4.3 (logistic-normal) and Appendix B.2 (Dirichlet).
Architecture reference: Section 6.3 / Appendix B.1, Table 6.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from rlte.models.distributions import LogisticNormalTransform


def _orthogonal_init(layer: nn.Linear, gain: float, bias_value: float | torch.Tensor = 0.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if isinstance(bias_value, torch.Tensor):
        with torch.no_grad():
            layer.bias.copy_(bias_value)
    else:
        nn.init.constant_(layer.bias, bias_value)


class _TrunkMLP(nn.Module):
    """Shared 2-hidden-layer tanh MLP trunk (128 units), Appendix B.1 / Table 6."""

    def __init__(self, state_dim: int, hidden_units: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        # Non-final layers: orthogonal init, gain 0.01 (Appendix B.1).
        _orthogonal_init(self.fc1, gain=0.01, bias_value=0.0)
        _orthogonal_init(self.fc2, gain=0.01, bias_value=0.0)
        self.act = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(state))
        h = self.act(self.fc2(h))
        return h


class LogisticNormalPolicy(nn.Module):
    """Primary (LN) policy: outputs mean mu of underlying Normal(mu, Sigma);
    actions are obtained via the logistic-normal transform h(x).

    Paper reference: Section 4.2, 4.3, Algorithm 1, Appendix B.1.
    """

    def __init__(self, state_dim: int, K: int, hidden_units: int = 128):
        """
        Args:
            state_dim: dimension of the normalized state feature vector.
            K: simplex dimension parameter (paper default K=6, Section 6.3).
            hidden_units: hidden layer width (paper default 128).
        """
        super().__init__()
        self.K = K
        self.trunk = _TrunkMLP(state_dim, hidden_units)
        self.mu_head = nn.Linear(hidden_units, K)
        # Final policy layer: orthogonal init gain 1e-5, bias = (-1,...,-1) in R^K
        # so that E[log(a_K/a_k)] ~ 1 at initialization (Eq. 16), i.e. the
        # "no-order" action a_K is initially favored.
        bias_init = torch.full((K,), -1.0)
        _orthogonal_init(self.mu_head, gain=1e-5, bias_value=bias_init)
        self.transform = LogisticNormalTransform()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return mu_theta(s): [B, K]."""
        assert state.dim() == 2, f"Expected [B, state_dim], got {tuple(state.shape)}"
        h = self.trunk(state)
        mu = self.mu_head(h)
        return mu

    def sample(self, state: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample x ~ N(mu, sigma*I_K) and transform to action a in S^K.

        Returns:
            a: [B, K+1] action on the simplex
            x: [B, K] raw normal sample (needed for log_prob during training)
            mu: [B, K] policy mean
        """
        mu = self.forward(state)
        std = math.sqrt(sigma)
        eps = torch.randn_like(mu)
        x = mu + std * eps
        a = self.transform.forward(x)
        return a, x, mu

    def log_prob(self, mu: torch.Tensor, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """log phi_theta(x | s) under N(mu, sigma*I_K), used in policy loss Eq. 14.

        Note: the logistic-normal density's Jacobian normalizing term does not
        depend on theta (Section 4.2, Eq. 8), so the policy gradient only
        requires the log-density of the underlying *normal* variable x.
        """
        var = torch.as_tensor(sigma, dtype=mu.dtype, device=mu.device)
        k = mu.shape[-1]
        diff = x - mu
        log_prob = -0.5 * (diff.pow(2).sum(dim=-1) / var
                            + k * torch.log(2 * math.pi * var))
        return log_prob

    @torch.no_grad()
    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluation-time deterministic action: a = h(mu_theta(s)) (Section 6.3)."""
        mu = self.forward(state)
        return self.transform.forward(mu)

    def __repr__(self) -> str:
        return f"LogisticNormalPolicy(K={self.K})"


class DirichletPolicy(nn.Module):
    """Benchmark (DR) policy: outputs Dirichlet concentration parameters alpha.

    Paper reference: Appendix B.2.
    """

    def __init__(self, state_dim: int, K: int, hidden_units: int = 128):
        super().__init__()
        self.K = K  # action dim is K+1 (alpha has K+1 components)
        self.trunk = _TrunkMLP(state_dim, hidden_units)
        self.alpha_head = nn.Linear(hidden_units, K + 1)
        self.softplus = nn.Softplus()
        # bias = softplus^-1(1,...,1,10) so that alpha ~ (1,...,1,10) at init
        # (the "no-order" action a_K is favored, mirroring the LN init trick).
        target = torch.cat([torch.ones(K), torch.tensor([10.0])])
        bias_init = torch.log(torch.expm1(target))  # softplus^-1
        _orthogonal_init(self.alpha_head, gain=1e-5, bias_value=bias_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return alpha(s): [B, K+1], strictly positive via softplus."""
        h = self.trunk(state)
        alpha = self.softplus(self.alpha_head(h)) + 1e-4
        return alpha

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.forward(state)
        dist = torch.distributions.Dirichlet(alpha)
        a = dist.rsample()
        return a, alpha

    def log_prob(self, alpha: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Dirichlet(alpha)
        return dist.log_prob(a.clamp_min(1e-8))

    @torch.no_grad()
    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """E[a] = alpha / sum(alpha), Eq. 17."""
        alpha = self.forward(state)
        return alpha / alpha.sum(dim=-1, keepdim=True)

    def __repr__(self) -> str:
        return f"DirichletPolicy(K={self.K})"
