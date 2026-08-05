"""
models/artemis.py
====================
Top-level ARTEMIS model assembling all 4 modules + prediction head + total
loss (Section 4.6, Algorithm 1).

Two-phase training (Appendix A.6.4, Algorithm 1):
    Phase 1 (Pretraining): encoder + SDE + prediction head trained jointly
        under L_total = L_forecast + lambda1*L_PDE + lambda2*L_MPR + lambda3*L_consist.
        Symbolic layer is NOT used (Algorithm 1, line 3 comment).
    Phase 2 (Distillation): everything frozen except the symbolic layer,
        trained via teacher-student distillation loss L_distill (Eq. 14).
    Conformal calibration: a third, non-gradient post-training step
        (Algorithm 1, lines 36-38).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from arxivist_artemis.models.laplace_neural_operator import LaplaceNeuralOperator
from arxivist_artemis.models.neural_sde import NeuralSDE, DriftNet, DiffusionNet
from arxivist_artemis.models.physics_losses import PhysicsLosses, AuxiliaryPricingNet
from arxivist_artemis.models.symbolic_bottleneck import SymbolicBottleneck, BasisLibrary
from arxivist_artemis.models.conformal_allocation import AdaptiveConformalPredictor


class ARTEMIS(nn.Module):
    """
    Full ARTEMIS architecture (Figure 1 / Section 4.6).

    Args:
        d_x: input feature dimension (dataset-dependent: 79/7/4/85).
        d_z: latent dimension. ASSUMED=64 per Figure 3 caption (SIR conf. 0.35).
        d_w: Wiener dimension. ASSUMED, ideally == d_z so market-price-of-risk
            (which requires an invertible square sigma) is well-defined.
        n_sde_steps: Euler-Maruyama step count M. ASSUMED=100 per Figure 2
            caption (SIR conf. 0.3).
        use_symbolic: whether to build the Module 3 symbolic bottleneck.
        use_conformal: whether to build the Module 4 conformal predictor
            (interval prediction only; the optional Kelly-QP layer is
            separate and NOT included here -- see conformal_allocation.py).
    """

    def __init__(
        self,
        d_x: int,
        d_z: int = 64,
        d_w: int = 16,
        n_lno_poles: int = 16,
        n_fourier: int = 8,
        drift_hidden_dim: int = 128,
        diffusion_hidden_dim: int = 128,
        aux_pricing_hidden_dim: int = 128,
        n_sde_steps: int = 100,
        n_basis_channels: Optional[int] = None,
        basis_lags: Optional[list] = None,
        use_symbolic: bool = True,
        use_conformal: bool = True,
        conformal_window: int = 500,
        conformal_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_z, self.d_w, self.n_sde_steps = d_z, d_w, n_sde_steps

        self.encoder = LaplaceNeuralOperator(d_x, d_z, n_lno_poles, n_fourier)
        drift = DriftNet(d_z, drift_hidden_dim, n_fourier)
        diffusion = DiffusionNet(d_z, d_w, diffusion_hidden_dim, n_fourier)
        self.sde = NeuralSDE(d_z, d_w, drift, diffusion)
        self.aux_pricing_net = AuxiliaryPricingNet(d_z, aux_pricing_hidden_dim, n_fourier)

        # Prediction head: y_hat = w^T z_M + b (Eq. 10).
        self.pred_head = nn.Linear(d_z, 1)

        self.use_symbolic = use_symbolic
        if use_symbolic:
            n_channels = n_basis_channels or d_x
            self.basis_library = BasisLibrary(n_channels, basis_lags)
            self.symbolic = SymbolicBottleneck(self.basis_library.n_basis)
        else:
            self.basis_library = None
            self.symbolic = None

        self.use_conformal = use_conformal
        if use_conformal:
            self.conformal = AdaptiveConformalPredictor(conformal_window, conformal_alpha)
        else:
            self.conformal = None

    def forward_pretrain(self, x: torch.Tensor, t: torch.Tensor, eval_times: torch.Tensor) -> dict:
        """
        Phase 1 forward pass (Algorithm 1, lines 6-14). Symbolic layer bypassed.

        Args:
            x: [B, N, d_x] irregular observations.
            t: [B, N] observation times.
            eval_times: [M] regular grid for the SDE solver.
        Returns:
            dict with 'y_hat' [B], 'z_traj' [B, n_sde_steps+1, d_z] (SDE-simulated),
            'z_enc' [B, M, d_z] (encoder outputs at grid points, for consistency loss).
        """
        z_enc = self.encoder(x, t, eval_times)  # [B, M, d_z]
        z0 = z_enc[:, 0, :]  # z_0 = E(x)(0)
        dt = float(eval_times[1] - eval_times[0]) if eval_times.shape[0] > 1 else 1.0
        z_traj = self.sde.simulate(z0, self.n_sde_steps, dt)  # [B, n_sde_steps+1, d_z]
        z_M = z_traj[:, -1, :]  # final latent state
        y_hat = self.pred_head(z_M).squeeze(-1)  # [B]
        return {"y_hat": y_hat, "z_traj": z_traj, "z_enc": z_enc, "z0": z0}

    def compute_losses(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        z_traj: torch.Tensor,
        z_enc: torch.Tensor,
        lambdas: dict,
        n_collocation: int = 32,
        task: str = "regression",
    ) -> dict:
        """
        Composite loss (Eq. 21, Section 4.5):
            L_total = L_forecast + lambda1*L_PDE + lambda2*L_MPR + lambda3*L_consist

        Args:
            y_hat, y_true: [B] predictions and targets.
            z_traj: [B, n_sde_steps+1, d_z] SDE-simulated trajectory.
            z_enc: [B, M, d_z] encoder outputs at grid points.
            lambdas: dict with keys 'pde','mpr','consist' (loss weights;
                ASSUMED, see config.yaml -- paper gives no numeric values).
            n_collocation: number of random collocation points sampled from
                the trajectory for the PDE loss.
            task: 'regression' (MSE) or 'classification' (BCE-with-logits),
                per Section 5.1 ("MSE for regression tasks ... and binary
                cross-entropy with logits for the DSLOB classification task").
        Returns:
            dict of scalar losses: forecast, pde, mpr, consist, total.
        """
        if task == "classification":
            l_forecast = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y_true.float())
        else:
            l_forecast = torch.nn.functional.mse_loss(y_hat, y_true.float())

        B, T, d_z = z_traj.shape
        idx = torch.randint(0, T, (min(n_collocation, B),), device=z_traj.device)
        batch_idx = torch.randint(0, B, (min(n_collocation, B),), device=z_traj.device)
        z_coll = z_traj[batch_idx, idx]  # [n_coll, d_z]
        t_coll = idx.float() / max(T - 1, 1)

        residuals = PhysicsLosses.feynman_kac_residual(
            self.aux_pricing_net, self.sde.drift, self.sde.diffusion, z_coll, t_coll
        )
        l_pde = PhysicsLosses.pde_loss(residuals)

        mu_coll = self.sde.drift(z_coll, t_coll)
        sigma_coll = self.sde.diffusion(z_coll, t_coll)
        if sigma_coll.shape[-1] == sigma_coll.shape[-2]:
            lam = PhysicsLosses.market_price_of_risk(mu_coll, sigma_coll)
            l_mpr = PhysicsLosses.mpr_loss(lam, kappa=2.0)
        else:
            l_mpr = torch.tensor(0.0, device=z_traj.device)

        M = z_enc.shape[1]
        z_sde_at_grid = z_traj[:, :: max(T // M, 1), :][:, :M, :]
        l_consist = torch.nn.functional.mse_loss(z_sde_at_grid, z_enc)

        total = (
            l_forecast
            + lambdas.get("pde", 1.0) * l_pde
            + lambdas.get("mpr", 1.0) * l_mpr
            + lambdas.get("consist", 1.0) * l_consist
        )
        return {"forecast": l_forecast, "pde": l_pde, "mpr": l_mpr, "consist": l_consist, "total": total}

    def forward_distill(self, x_window: torch.Tensor) -> torch.Tensor:
        """
        Phase 2 symbolic-layer forward pass (Algorithm 1, lines 27-34).

        Args:
            x_window: [B, L, C] raw input window fed to the basis library.
        Returns:
            [B] symbolic predictions y_hat_symb.
        """
        assert self.use_symbolic, "ARTEMIS was constructed with use_symbolic=False"
        basis_features = self.basis_library.compute_features(x_window)
        return self.symbolic(basis_features)
