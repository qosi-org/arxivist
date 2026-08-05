"""
models/physics_losses.py
==========================
Feynman-Kac PDE residual (Section 4.2.1) and market-price-of-risk penalty
(Section 4.2.2), Appendix A.4.

IMPORTANT CAVEAT (documented per architecture_plan.json risk assessment):
The paper's own Appendix A.4.1 explicitly works through why penalizing this
residual to zero over-constrains the market price of risk to zero (i.e., no
risk premium), and concludes that using a single auxiliary network V_psi
"is not sufficient for full no-arbitrage, [but] provides a useful
regularizer." This is the paper's own admission, not an ArXivist critique.
The implementation below follows Algorithm 1 / Eq. (7) exactly as described
-- this is what the ablation study (Table 3) actually measures the effect
of, and it demonstrably matters a great deal for directional accuracy, even
though it does not provably enforce true no-arbitrage in the strict sense.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _TimeEmbed(nn.Module):
    def __init__(self, n_frequencies: int = 8) -> None:
        super().__init__()
        self.freqs = nn.Parameter(torch.logspace(-2, 2, n_frequencies))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * t.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class AuxiliaryPricingNet(nn.Module):
    """
    V_psi(z,t): auxiliary neural network representing a generic derivative
    pricing function (Section 4.2.1). Architecture (depth/width) is NOT
    specified in the paper -- ASSUMED as a 2-layer MLP matching the
    drift/diffusion networks' scale (SIR confidence low; documented gap).
    """

    def __init__(self, d_z: int, hidden_dim: int = 128, n_fourier: int = 8) -> None:
        super().__init__()
        self.time_embed = _TimeEmbed(n_fourier)
        self.net = nn.Sequential(
            nn.Linear(d_z + 2 * n_fourier, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """z: [N, d_z], t: [N] -> V: [N]"""
        te = self.time_embed(t)
        return self.net(torch.cat([z, te], dim=-1)).squeeze(-1)


class PhysicsLosses:
    """
    Stateless collection of the two physics-informed regularization losses.
    """

    @staticmethod
    def feynman_kac_residual(
        V: AuxiliaryPricingNet,
        drift_fn,
        diffusion_fn,
        z: torch.Tensor,
        t: torch.Tensor,
        r: float = 0.0,
    ) -> torch.Tensor:
        """
        R_FK(z,t) = dV/dt + mu.grad_z(V) + 0.5*tr(sigma sigma^T grad_z^2(V)) - r*V
        (Section 4.2.1, Appendix A.4.1, r=0 "for simplicity" per the paper).

        Args:
            V: the auxiliary pricing network.
            drift_fn: callable (z,t) -> mu, [N, d_z].
            diffusion_fn: callable (z,t) -> sigma, [N, d_z, d_w].
            z: [N, d_z] collocation points, requires_grad will be set internally.
            t: [N] collocation times, requires_grad will be set internally.
            r: risk-free rate, set to 0 per paper's stated simplification.
        Returns:
            [N] PDE residuals R_FK(z_i, t_i).
        """
        z = z.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)

        Vz = V(z, t)  # [N]
        grad_V = torch.autograd.grad(Vz.sum(), [z, t], create_graph=True)
        grad_z_V, dV_dt = grad_V[0], grad_V[1]  # [N, d_z], [N]

        d_z = z.shape[-1]
        # Hessian w.r.t. z: [N, d_z, d_z], computed row-by-row via autograd.
        hessian_rows = []
        for i in range(d_z):
            grad_i = torch.autograd.grad(
                grad_z_V[:, i].sum(), z, create_graph=True, retain_graph=True
            )[0]  # [N, d_z]
            hessian_rows.append(grad_i)
        hessian_z_V = torch.stack(hessian_rows, dim=1)  # [N, d_z, d_z]

        mu = drift_fn(z, t)  # [N, d_z]
        sigma = diffusion_fn(z, t)  # [N, d_z, d_w]
        sigma_sigma_T = torch.einsum("nij,nkj->nik", sigma, sigma)  # [N, d_z, d_z]

        drift_term = (mu * grad_z_V).sum(-1)  # [N]
        trace_term = 0.5 * torch.einsum("nij,nij->n", sigma_sigma_T, hessian_z_V)  # [N]

        return dV_dt + drift_term + trace_term - r * Vz

    @staticmethod
    def pde_loss(residuals: torch.Tensor) -> torch.Tensor:
        """L_PDE = mean(R_FK^2) (Eq. 7)."""
        return (residuals ** 2).mean()

    @staticmethod
    def market_price_of_risk(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        lambda(t) = sigma^{-1} @ mu (Eq. 8). Uses torch.linalg.solve rather
        than an explicit matrix inverse for numerical stability. Requires
        sigma to be square (d_z == d_w) and invertible; DiffusionNet's
        Cholesky-style construction guarantees invertibility when d_z==d_w.

        Args:
            mu: [N, d_z] drift values.
            sigma: [N, d_z, d_w] diffusion values (d_z must equal d_w here).
        Returns:
            [N, d_w] market price of risk vectors.
        """
        assert sigma.shape[-1] == sigma.shape[-2], (
            "market_price_of_risk requires square sigma (d_z == d_w); "
            f"got sigma shape {tuple(sigma.shape)}. See config.yaml model.artemis.d_w."
        )
        return torch.linalg.solve(sigma, mu.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def mpr_loss(lam: torch.Tensor, kappa: float = 2.0) -> torch.Tensor:
        """
        L_MPR = mean(max(0, ||lambda||^2 - kappa^2)) (Eq. 9).

        Args:
            lam: [N, d_w] market price of risk vectors.
            kappa: Sharpe-ratio threshold. Paper's Appendix A.4.2 suggests
                kappa=2 "for daily data" as "a reasonable choice" -- ASSUMED
                as a default here (SIR confidence 0.4), exposed as
                config.model.artemis.mpr_kappa.
        Returns:
            scalar loss.
        """
        sharpe_sq = (lam ** 2).sum(-1)  # [N]
        return torch.relu(sharpe_sq - kappa ** 2).mean()
