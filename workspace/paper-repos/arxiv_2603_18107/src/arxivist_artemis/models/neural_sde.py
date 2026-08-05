"""
models/neural_sde.py
======================
Module 2: economics-informed latent dynamics via a neural stochastic
differential equation (Section 4.2, Appendix A.3):

    dz(t) = mu_theta(z(t),t) dt + sigma_phi(z(t),t) dW(t)

simulated via Euler-Maruyama (Eq. 6), with gradients flowing through the
solver via the reparametrisation trick (noise sampled independently of
parameters).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _TimeEmbed(nn.Module):
    """Shared small Fourier time embedding for drift/diffusion nets (matches LNO's, but independent params)."""

    def __init__(self, n_frequencies: int = 8) -> None:
        super().__init__()
        self.freqs = nn.Parameter(torch.logspace(-2, 2, n_frequencies))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * t.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class DriftNet(nn.Module):
    """
    mu_theta(z,t): 1-hidden-layer MLP with tanh activation (Appendix A.3.2):
        mu_theta(z,t) = W2*tanh(W1*[z; TimeEmbedding(t)] + b1) + b2
    """

    def __init__(self, d_z: int, hidden_dim: int = 128, n_fourier: int = 8) -> None:
        super().__init__()
        self.time_embed = _TimeEmbed(n_fourier)
        self.net = nn.Sequential(
            nn.Linear(d_z + 2 * n_fourier, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d_z),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """z: [..., d_z], t: [...] -> mu: [..., d_z]"""
        te = self.time_embed(t)
        return self.net(torch.cat([z, te], dim=-1))


class DiffusionNet(nn.Module):
    """
    sigma_phi(z,t) = L_phi @ D_phi (Appendix A.3.2), Cholesky-style
    factorization guaranteeing sigma_phi invertible and sigma*sigma^T PSD:
        L_phi = Tril(MLP_phiL(...)) + I   (unit-diagonal lower triangular)
        D_phi = diag(Softplus(MLP_phiD(...)))   (strictly positive diagonal)
    """

    def __init__(self, d_z: int, d_w: int, hidden_dim: int = 128, n_fourier: int = 8) -> None:
        super().__init__()
        self.d_z, self.d_w = d_z, d_w
        self.time_embed = _TimeEmbed(n_fourier)
        in_dim = d_z + 2 * n_fourier
        n_tril = d_z * d_w
        self.net_L = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_tril))
        self.net_D = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, d_w))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """z: [..., d_z], t: [...] -> sigma: [..., d_z, d_w]"""
        te = self.time_embed(t)
        inp = torch.cat([z, te], dim=-1)
        L_flat = self.net_L(inp)  # [..., d_z*d_w]
        L = L_flat.view(*z.shape[:-1], self.d_z, self.d_w)
        min_dim = min(self.d_z, self.d_w)
        tril_mask = torch.tril(torch.ones(self.d_z, self.d_w, device=z.device))
        L = L * tril_mask
        eye = torch.zeros(self.d_z, self.d_w, device=z.device)
        eye[:min_dim, :min_dim] = torch.eye(min_dim, device=z.device)
        L = L + eye
        D = torch.nn.functional.softplus(self.net_D(inp))  # [..., d_w], strictly positive
        return L * D.unsqueeze(-2)  # [..., d_z, d_w], equivalent to L @ diag(D)


class NeuralSDE(nn.Module):
    """
    Wraps drift + diffusion networks for Euler-Maruyama simulation of the
    latent SDE (Eq. 5-6).

    Args:
        d_z: latent dimension.
        d_w: Wiener process dimension. ASSUMED, unspecified in paper.
        drift: a DriftNet instance.
        diffusion: a DiffusionNet instance.
    """

    def __init__(self, d_z: int, d_w: int, drift: DriftNet, diffusion: DiffusionNet) -> None:
        super().__init__()
        self.d_z, self.d_w = d_z, d_w
        self.drift = drift
        self.diffusion = diffusion

    def simulate(self, z0: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
        """
        Euler-Maruyama simulation (Eq. 6):
            z_{j+1} = z_j + mu_theta(z_j,t_j)*dt + sigma_phi(z_j,t_j)*sqrt(dt)*eps_j

        Args:
            z0: [B, d_z] initial latent state (from the LNO encoder at t=0).
            n_steps: number of simulation steps M. ASSUMED=100 per Figure 2's
                caption hint (SIR confidence 0.3), exposed as a config param.
            dt: Euler-Maruyama step size.
        Returns:
            [B, n_steps+1, d_z] full simulated trajectory (including z0).
        """
        B = z0.shape[0]
        device = z0.device
        traj = [z0]
        z = z0
        for j in range(n_steps):
            t_j = torch.full((B,), j * dt, device=device)
            mu = self.drift(z, t_j)  # [B, d_z]
            sigma = self.diffusion(z, t_j)  # [B, d_z, d_w]
            eps = torch.randn(B, self.d_w, device=device)  # reparametrisation: independent of params
            diffusion_term = torch.einsum("bzw,bw->bz", sigma, eps) * math.sqrt(dt)
            z = z + mu * dt + diffusion_term
            traj.append(z)
        return torch.stack(traj, dim=1)  # [B, n_steps+1, d_z]
