"""
Loss functions for all three experiments.

Implements:
  1. EnergyScore     — GSL experiment (Section 5.1, Appendix E.4, Eq. 10)
  2. KLDivCategorical— VAE experiment, analytic KL (Appendix F.4)
  3. VAELoss         — VAE experiment, ELBO = recon + KL (Appendix F.4, Eq. 11)

Paper: "Beyond Softmax: A Natural Parameterization for Categorical Random Variables"
arXiv: 2509.24728v2. ICML 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EnergyScore(nn.Module):
    """Multivariate Energy Score (ES) loss for the GSL experiment.

    A proper scoring rule (multivariate extension of CRPS):

        ES = (1/M) Σ_m ||f_ψ(x, A_m) - y||₂
           - (1/(2M(M-1))) Σ_{m≠n} ||f_ψ(x, A_m) - f_ψ(x, A_n)||₂

    Lower is better. The second term rewards diversity among predictions,
    which calibrates the latent graph distribution.

    Paper: Appendix E.4, Eq. 10. Gneiting & Raftery (2007). Confidence: 0.95.

    Notes:
        The second term is O(M²) in pairwise distances. Implemented via
        broadcasting (RISK-08); at M=32 this is 32*31=992 pairs, manageable on GPU.
    """

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute the Energy Score.

        Args:
            preds:  Model predictions for M graph samples, shape [M, B, D].
            target: Ground-truth outputs, shape [B, D].

        Returns:
            Scalar Energy Score loss (mean over batch).
        """
        assert preds.dim() == 3, (
            f"EnergyScore expects preds of shape [M, B, D], got {preds.shape}"
        )
        assert target.dim() == 2, (
            f"EnergyScore expects target of shape [B, D], got {target.shape}"
        )
        M, B, D = preds.shape

        # Eq. 10, term 1: (1/M) Σ_m ||f_ψ(x, A_m) - y||₂
        # preds: [M, B, D], target: [B, D] → broadcast [M, B, D]
        term1 = torch.norm(preds - target.unsqueeze(0), dim=-1).mean(dim=0)  # [B]

        # Eq. 10, term 2: (1/(2M(M-1))) Σ_{m≠n} ||f_ψ(x,A_m) - f_ψ(x,A_n)||₂
        # Vectorized pairwise distances via broadcasting: [M, 1, B, D] - [1, M, B, D]
        preds_i = preds.unsqueeze(1)   # [M, 1, B, D]
        preds_j = preds.unsqueeze(0)   # [1, M, B, D]
        pairwise = torch.norm(preds_i - preds_j, dim=-1)  # [M, M, B]
        # Exclude diagonal (m==n)
        mask = 1.0 - torch.eye(M, device=preds.device).unsqueeze(-1)  # [M, M, 1]
        term2 = (pairwise * mask).sum(dim=[0, 1]) / (2.0 * M * (M - 1))  # [B]

        es = (term1 - term2).mean()  # scalar
        return es


class KLDivCategorical(nn.Module):
    """Analytic KL divergence between a categorical posterior and a uniform prior.

    KL(Cat(q) || Uniform(1/K)) = Σ_k q_k * log(q_k * K)
                                = Σ_k q_k * (log q_k + log K)
                                = -H(q) + log K

    where H(q) = -Σ_k q_k * log q_k is the entropy of q.

    Summed over N independent categorical variables, averaged over batch.

    Paper: Appendix F.4. Confidence: 0.98 (explicitly stated).
    """

    def forward(self, q_probs: Tensor) -> Tensor:
        """Compute KL(Cat(q) || Uniform(1/K)) analytically.

        Args:
            q_probs: Posterior probabilities, shape [B, N, K].
                     N = number of categorical variables, K = number of classes.

        Returns:
            Scalar KL divergence (sum over N, mean over B).
        """
        assert q_probs.dim() == 3, (
            f"KLDivCategorical expects q_probs of shape [B, N, K], got {q_probs.shape}"
        )
        B, N, K = q_probs.shape
        eps = 1e-8

        # KL(Cat(q) || Uniform) = Σ_k q_k * log(q_k * K)
        kl_per_var = (q_probs * (torch.log(q_probs.clamp(min=eps)) + torch.log(
            torch.tensor(float(K), device=q_probs.device)
        ))).sum(dim=-1)   # [B, N]

        # Sum over N variables, mean over batch
        return kl_per_var.sum(dim=-1).mean()   # scalar


class VAELoss(nn.Module):
    """ELBO loss for the categorical VAE experiment.

    ELBO = -E_{q(C|x)}[log p(x|C)] + KL(q(C|x) || p(C))
         = reconstruction_loss + kl_divergence

    Reconstruction loss: binary cross-entropy between input and reconstruction.
    KL term: analytic KL(Cat(q) || Uniform) computed by KLDivCategorical.

    Paper: Appendix F.4, Eq. 11. Kingma & Welling (2014). Confidence: 0.98.

    Args:
        beta: Weight on the KL term. Default 1.0 (standard ELBO).
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self._kl = KLDivCategorical()

    def forward(
        self,
        x_recon: Tensor,
        x_target: Tensor,
        q_probs: Tensor,
    ) -> dict:
        """Compute ELBO components.

        Args:
            x_recon:  Reconstructed image (sigmoid output), shape [B, 1, H, W].
            x_target: Original input image, shape [B, 1, H, W].
            q_probs:  Posterior probabilities, shape [B, N, K].

        Returns:
            Dict with keys:
              'total':  total loss (scalar, to be minimized)
              'recon':  reconstruction loss (scalar)
              'kl':     KL divergence (scalar)
        """
        assert x_recon.shape == x_target.shape, (
            f"Shape mismatch: x_recon {x_recon.shape} vs x_target {x_target.shape}"
        )

        # Reconstruction: binary cross-entropy averaged over pixels and batch
        # Paper: Appendix F.4 — "binary cross-entropy between the input and the output"
        recon = F.binary_cross_entropy(x_recon, x_target, reduction="mean")

        # KL divergence: analytic
        kl = self._kl(q_probs)

        total = recon + self.beta * kl
        return {"total": total, "recon": recon, "kl": kl}

    def __repr__(self) -> str:
        return f"VAELoss(beta={self.beta})"
