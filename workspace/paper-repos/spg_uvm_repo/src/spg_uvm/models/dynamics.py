"""
Multi-asset log-Euler simulation scheme for the Uncertain Volatility Model.

Implements the discrete-time transition function F from Section 2.3:

    F(x, a, xi) = x * exp((r - 0.5*diag(aa^T)) * dt + a * sqrt(dt) * xi)

where xi ~ N(0, I_d), a = diag(sigma) * L is the volatility matrix,
and dt = T/N is the time step size.

Reference: Eq. (8), Section 2.3, arXiv:2605.06670.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LogEulerScheme(nn.Module):
    """
    Log-Euler (log-normal Euler-Maruyama) scheme for multi-asset GBM.

    Implements Eq. (8) from arXiv:2605.06670:
        F(x, a, xi) = x ⊙ exp[(r - 1/2 * diag(aa^T)) * T/N + a * sqrt(T/N) * xi]

    The volatility matrix a = diag(sigma) * L where L is the lower-triangular
    Cholesky factor from the C-vine parameterization.

    Args:
        d:   Number of assets.
        T:   Time to maturity.
        N:   Number of time steps.
        r:   Risk-free rate.
    """

    def __init__(self, d: int, T: float, N: int, r: float) -> None:
        super().__init__()
        self.d = d
        self.T = T
        self.N = N
        self.r = r
        self.dt = T / N
        self.sqrt_dt = (T / N) ** 0.5

    def step(self, x: Tensor, a: Tensor, xi: Tensor) -> Tensor:
        """
        Apply one log-Euler step.

        Eq. (8) of arXiv:2605.06670.

        Args:
            x:  Asset prices at time n, shape [B, d]. Must be > 0.
            a:  Volatility matrix (Cholesky factor scaled), shape [B, d, d].
                a = diag(sigma) * L where L is from C-vine.
            xi: Standard Gaussian noise, shape [B, d].

        Returns:
            x_next: Asset prices at time n+1, shape [B, d].
        """
        assert x.dim() == 2 and x.shape[1] == self.d, f"x: expected [B,{self.d}], got {x.shape}"
        assert a.dim() == 3 and a.shape[1:] == (self.d, self.d), (
            f"a: expected [B,{self.d},{self.d}], got {a.shape}"
        )
        assert xi.dim() == 2 and xi.shape[1] == self.d, f"xi: expected [B,{self.d}], got {xi.shape}"

        # Instantaneous covariance: aa^T, shape [B, d, d]
        aa_T = torch.bmm(a, a.transpose(1, 2))  # [B, d, d]

        # Diagonal of aa^T = instantaneous variance per asset, shape [B, d]
        diag_aaT = aa_T.diagonal(dim1=-2, dim2=-1)  # [B, d]

        # Drift term: (r - 0.5 * sigma_i^2) * dt, shape [B, d]
        drift = (self.r - 0.5 * diag_aaT) * self.dt  # [B, d]

        # Diffusion term: a * sqrt(dt) * xi, shape [B, d]
        # a @ xi: [B, d, d] @ [B, d, 1] -> [B, d, 1] -> [B, d]
        diffusion = torch.bmm(a, xi.unsqueeze(-1)).squeeze(-1) * self.sqrt_dt  # [B, d]

        # Log-normal step: X_{n+1} = X_n * exp(drift + diffusion)
        x_next = x * torch.exp(drift + diffusion)  # [B, d]
        return x_next

    def build_action_matrix(self, sigma: Tensor, L: Tensor) -> Tensor:
        """
        Build the volatility matrix a = diag(sigma) * L.

        In the log-Euler scheme, the Brownian increment is a * sqrt(dt) * xi,
        so a must satisfy aa^T = diag(sigma) * rho * diag(sigma).

        The C-vine gives L with rho = LL^T, so:
            a = diag(sigma) * L

        Args:
            sigma: Volatility vector [B, d].
            L:     Cholesky factor from C-vine [B, d, d].

        Returns:
            a: Volatility matrix [B, d, d].
        """
        # diag(sigma): [B, d, d] — each batch element is a diagonal matrix
        diag_sigma = torch.diag_embed(sigma)  # [B, d, d]
        # a = diag(sigma) @ L [B, d, d]
        a = torch.bmm(diag_sigma, L)
        return a

    def __repr__(self) -> str:
        return f"LogEulerScheme(d={self.d}, T={self.T}, N={self.N}, r={self.r})"
