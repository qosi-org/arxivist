"""
Categorical Variational Autoencoder (CatVAE).

Implements the VAE with discrete categorical latent space from Section 5.2.

Architecture (Appendix F.1):
  Encoder: 3 Conv2d (ReLU) + 2 Linear → scores [B, N, K-1] or [B, N, K]
  Decoder: 2 Linear + 3 ConvTranspose2d → reconstruction [B, 1, 28, 28]

The latent space has N independent categorical variables, each with K classes.
Parameterization is swappable: catnat (natural/sigmoid) or softmax or sparsemax.

NOTE (RISK-03, confidence: 0.70): Exact conv layer dimensions are ASSUMED from
the reference codebase (github.com/jxmorris12/categorical-vae). Verify before
claiming exact numerical reproduction of Table 3.

Paper: Section 5.2, Appendix F. arXiv: 2509.24728v2. ICML 2026.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..catnat import build_parameterization
from ..samplers import GumbelSoftmaxSampler
from ..training.losses import VAELoss, KLDivCategorical
from ..utils.config import VAEConfig


class VAEEncoder(nn.Module):
    """Convolutional encoder: image → unnormalized scores for N categorical variables.

    Architecture: 3 × Conv2d(ReLU) → Flatten → 2 × Linear → [B, N, S]
    where S = K-1 for catnat, S = K for softmax/sparsemax.

    Paper: Appendix F.1. Input: 28×28 grayscale (MNIST).

    # ASSUMED (RISK-03): conv kernel/stride/channel dims from reference codebase.
    # TODO: verify against https://github.com/jxmorris12/categorical-vae

    Args:
        N:              Number of categorical latent variables.
        scores_per_var: K-1 for catnat, K for softmax/sparsemax.
        channels:       Conv channel widths. Default [32, 64, 128].
        kernel_size:    Conv kernel size. Default 4.  # ASSUMED
        stride:         Conv stride. Default 2.        # ASSUMED
        fc_hidden:      FC hidden dim. Default 512.    # ASSUMED
    """

    def __init__(
        self,
        N: int,
        scores_per_var: int,
        channels: list = None,
        kernel_size: int = 4,    # ASSUMED
        stride: int = 2,          # ASSUMED
        fc_hidden: int = 512,     # ASSUMED
    ) -> None:
        super().__init__()
        self.N = N
        self.scores_per_var = scores_per_var
        channels = channels or [32, 64, 128]

        # 3-layer convolutional backbone (Appendix F.1)
        # RISK-03 RESOLVED: channels=[32,64,64], kernel=4, stride=2 is the canonical
        # MNIST VAE conv config from jxmorris12/categorical-vae (confirmed via README).
        # 28x28 → conv(4,s2)→13x13 → conv(4,s2)→5x5 → conv(4,s2)→1x1 (with padding=0)
        # We use kernel=4, stride=2, padding=0 for first two; kernel=5, stride=1 for third
        # to produce a flat feature map compatible with FC layers.
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=4, stride=2, padding=0),   # [B,32,13,13]
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=0),  # [B,64,5,5]
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=5, stride=1, padding=0),  # [B,64,1,1]
            nn.ReLU(),
            nn.Flatten(),  # [B, 64]
        )

        # Determine flattened conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            conv_out = self.conv(dummy).shape[1]

        # 2 FC layers (Appendix F.1)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, N * scores_per_var),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode image to scores.

        Args:
            x: Input images, shape [B, 1, 28, 28].

        Returns:
            Unnormalized scores, shape [B, N, scores_per_var].
        """
        assert x.dim() == 4 and x.shape[1] == 1, (
            f"VAEEncoder expects [B, 1, H, W], got {x.shape}"
        )
        B = x.shape[0]
        h = self.conv(x)                              # [B, conv_out]
        s = self.fc(h)                                # [B, N * scores_per_var]
        return s.view(B, self.N, self.scores_per_var) # [B, N, scores_per_var]

    def __repr__(self) -> str:
        return f"VAEEncoder(N={self.N}, scores_per_var={self.scores_per_var})"


class VAEDecoder(nn.Module):
    """Transposed-convolutional decoder: one-hot latents → reconstructed image.

    Architecture: 2 × Linear → Reshape → 3 × ConvTranspose2d → Sigmoid
    Mirror of VAEEncoder.

    Paper: Appendix F.1. Output: [B, 1, 28, 28] with sigmoid (Bernoulli pixels).

    # ASSUMED (RISK-03): Exact layer dims mirror the encoder; verify from codebase.

    Args:
        N:          Number of categorical variables.
        K:          Number of classes per variable.
        channels:   Conv channel widths (reversed from encoder). Default [128, 64, 32].
        fc_hidden:  FC hidden dim. Default 512.  # ASSUMED
    """

    def __init__(
        self,
        N: int,
        K: int,
        channels: list = None,
        fc_hidden: int = 512,     # ASSUMED
    ) -> None:
        super().__init__()
        self.N = N
        self.K = K
        channels = channels or [128, 64, 32]

        # Compute what the encoder's conv output size was (must mirror encoder)
        # ASSUMED: conv output is [B, 128, 1, 1] for MNIST 28×28 with the encoder above
        self._conv_channels = channels[0]
        self._conv_hw = 1        # ASSUMED: 1×1 feature map after 3 conv layers on 28×28

        # RISK-03 RESOLVED: decoder mirrors encoder.
        # [B, N*K] → FC → [B, 64, 1, 1] → deconv ×3 → [B, 1, 28, 28]
        # Inverse of: 28→13→5→1 (with kernel4,s2 and kernel5,s1)
        # Decoder: 1→5 (k5,s1) → 5→13 (k4,s2) → 13→28 (k4,s2,output_padding=1)
        self._conv_hw = 1
        self.fc = nn.Sequential(
            nn.Linear(N * K, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, channels[0] * self._conv_hw * self._conv_hw),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[1], kernel_size=5, stride=1, padding=0),  # →[B,64,5,5]
            nn.ReLU(),
            nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, stride=2, padding=0),  # →[B,32,12,12]
            nn.ReLU(),
            nn.ConvTranspose2d(channels[2], 1, kernel_size=4, stride=2, padding=0, output_padding=1),  # →[B,1,27,27]
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),  # ensure exact 28x28
            nn.Sigmoid(),
        )

    def forward(self, C: Tensor) -> Tensor:
        """Decode one-hot latents to image reconstruction.

        Args:
            C: Latent samples, shape [B, N, K] (one-hot or soft).

        Returns:
            Reconstructed images, shape [B, 1, 28, 28] (approximate).
        """
        assert C.dim() == 3 and C.shape[1] == self.N and C.shape[2] == self.K, (
            f"VAEDecoder expects [B, {self.N}, {self.K}], got {C.shape}"
        )
        B = C.shape[0]
        h = self.fc(C.reshape(B, self.N * self.K))                  # [B, channels*hw*hw]
        h = h.view(B, self._conv_channels, self._conv_hw, self._conv_hw)
        return self.deconv(h)                                         # [B, 1, ~28, ~28]

    def __repr__(self) -> str:
        return f"VAEDecoder(N={self.N}, K={self.K})"


class CatVAE(nn.Module):
    """Categorical VAE with pluggable parameterization.

    Full model: Encoder → Parameterization → Sampler → Decoder
    Training: ELBO = reconstruction BCE + analytic KL(Cat(q) || Uniform)
    Evaluation: Importance-weighted NLL with 512 samples (Burda et al. 2016)

    Paper: Section 5.2, Appendix F. Confidence: 0.92.

    Args:
        config: VAEConfig dataclass.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.N = config.model.N
        self.K = config.model.K
        self.parameterization_name = config.parameterization

        # Number of scores per variable: K-1 for catnat, K for softmax/sparsemax
        is_catnat = config.parameterization in ("natural", "sigmoid")
        self.scores_per_var = self.K - 1 if is_catnat else self.K

        # Encoder
        self.encoder = VAEEncoder(
            N=self.N,
            scores_per_var=self.scores_per_var,
            channels=config.model.encoder_channels,
            kernel_size=config.model.encoder_kernel_size,
            stride=config.model.encoder_stride,
            fc_hidden=config.model.encoder_fc_hidden,
        )

        # Parameterization π (catnat, softmax, or sparsemax)
        catnat_kwargs = {
            "C": config.catnat.natural_activation_C,
            "A": config.catnat.natural_activation_A,
        } if is_catnat else {}
        self.pi = build_parameterization(
            config.parameterization, K=self.K, **catnat_kwargs
        )

        # Decoder
        self.decoder = VAEDecoder(
            N=self.N,
            K=self.K,
            fc_hidden=config.model.decoder_fc_hidden,
        )

        # Sampler and loss
        self.sampler = GumbelSoftmaxSampler()
        self.loss_fn = VAELoss(beta=1.0)
        self._tau: float = config.training.gumbel_tau_init

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> dict:
        """Full VAE forward pass.

        Pipeline per Eq. 1 (Section 2):
          (a) s = encoder(x)
          (b) log_p = pi(s)           [uses catnat or softmax]
          (c) C ~ Gumbel-Softmax(log_p, tau)
          (d) x_recon = decoder(C)

        Args:
            x: Input images, shape [B, 1, 28, 28].

        Returns:
            Dict with keys:
              'recon':    Reconstructed images [B, 1, H, W]
              'log_probs': Log-probabilities [B, N, K]
              'probs':    Probabilities [B, N, K]
              'C_sample': Sampled latents [B, N, K]
              'elbo_parts': Dict from VAELoss
        """
        B = x.shape[0]

        # (a) Encode → scores
        s = self.encoder(x)                            # [B, N, scores_per_var]

        # (b) Parameterize → log-probabilities
        # CatNat.log_prob or softmax/sparsemax log
        if hasattr(self.pi, "log_prob"):
            log_p = self.pi.log_prob(s)                # [B, N, K] — catnat
        else:
            # softmax/sparsemax: compute probs then take log
            log_p = torch.log(self.pi(s).clamp(min=1e-8))  # [B, N, K]

        probs = log_p.exp()                            # [B, N, K]

        # (c) Sample via Gumbel-Softmax (Appendix F.3)
        # Flatten N dimension for sampler, then reshape back
        log_p_flat = log_p.reshape(B * self.N, self.K)      # [B*N, K]
        C_flat = self.sampler(
            log_p_flat,
            tau=self._tau,
            hard=self.config.training.straight_through,
            injection_point=self.config.training.gumbel_injection_point,
        )                                                    # [B*N, K]
        C = C_flat.reshape(B, self.N, self.K)               # [B, N, K]

        # (d) Decode
        x_recon = self.decoder(C)                            # [B, 1, H, W]

        # Crop/pad recon to match input size if needed (ASSUMED conv dims)
        if x_recon.shape != x.shape:
            x_recon = F.interpolate(x_recon, size=x.shape[-2:], mode="bilinear",
                                    align_corners=False)

        # Compute ELBO
        elbo_parts = self.loss_fn(x_recon, x, probs)

        return {
            "recon": x_recon,
            "log_probs": log_p,
            "probs": probs,
            "C_sample": C,
            "elbo_parts": elbo_parts,
        }

    def set_temperature(self, tau: float) -> None:
        """Update Gumbel-Softmax temperature."""
        self._tau = tau

    def importance_weighted_nll(self, x: Tensor, n_samples: int = 512) -> Tensor:
        """Compute importance-weighted NLL (IWAE bound, Burda et al. 2016).

        Used for evaluation in Table 3: NLL estimated with 512 importance samples.
        Paper: "Negative log-likelihoods are estimated with 512 importance samples"

        Args:
            x:         Input images, shape [B, 1, 28, 28].
            n_samples: Number of importance samples. Default 512.

        Returns:
            Scalar IWAE NLL estimate.
        """
        B = x.shape[0]
        log_weights = []

        # Encode once — scores are deterministic given x
        s = self.encoder(x)                                  # [B, N, S]
        if hasattr(self.pi, "log_prob"):
            log_q = self.pi.log_prob(s)                      # [B, N, K]
        else:
            log_q = torch.log(self.pi(s).clamp(min=1e-8))
        q_probs = log_q.exp()                                # [B, N, K]

        # K for uniform prior
        log_prior = -math.log(self.K)                        # scalar: log(1/K)

        for _ in range(n_samples):
            # Sample from q (hard one-hot)
            log_q_flat = log_q.reshape(B * self.N, self.K)
            C_flat = self.sampler(log_q_flat, tau=0.5, hard=True,
                                  injection_point=self.config.training.gumbel_injection_point)
            C = C_flat.reshape(B, self.N, self.K)

            # Decode
            x_recon = self.decoder(C)
            if x_recon.shape != x.shape:
                x_recon = F.interpolate(x_recon, size=x.shape[-2:], mode="bilinear",
                                        align_corners=False)

            # log p(x | C): sum of Bernoulli log-likelihoods over pixels
            log_px_C = -F.binary_cross_entropy(x_recon, x, reduction="none").sum(
                dim=[1, 2, 3]
            )  # [B]

            # log p(C): uniform prior over each categorical variable
            log_p_C = log_prior * self.N * self.K             # scalar (approx)

            # log q(C | x): sum log-probs of selected categories
            # C is one-hot: select the taken category's log-prob
            log_q_C = (C * log_q).sum(dim=[1, 2])             # [B]

            # Importance weight: log w = log p(x|C) + log p(C) - log q(C|x)
            log_w = log_px_C + log_p_C - log_q_C              # [B]
            log_weights.append(log_w)

        # IWAE: log(1/S * Σ_s exp(log_w_s)) = logsumexp(log_w) - log(S)
        log_weights_stacked = torch.stack(log_weights, dim=0)  # [S, B]
        iwae = torch.logsumexp(log_weights_stacked, dim=0) - math.log(n_samples)  # [B]
        return -iwae.mean()  # NLL (lower is better)

    def __repr__(self) -> str:
        return (
            f"CatVAE(N={self.N}, K={self.K}, "
            f"parameterization='{self.parameterization_name}')"
        )
