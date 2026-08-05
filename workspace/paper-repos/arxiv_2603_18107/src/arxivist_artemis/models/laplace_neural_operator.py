"""
models/laplace_neural_operator.py
===================================
Module 1: continuous-time encoder based on a Laplace Neural Operator
(Section 4.1, Appendix A.2). Maps irregularly sampled observations directly
to a continuous latent function without interpolation or regular resampling.

ASSUMED hyperparameters (never given numeric values in the paper):
    d_z (latent dim), n_poles (K), n_fourier (F) -- see config.yaml.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _TimeEmbedding(nn.Module):
    """
    Fourier time embedding with learnable frequencies (Appendix A.2.4):
        TimeEmbedding(t) = [sin(2*pi*f_1*t), cos(2*pi*f_1*t), ..., sin(2*pi*f_F*t), cos(2*pi*f_F*t)]
    """

    def __init__(self, n_frequencies: int = 8) -> None:
        super().__init__()
        init_freqs = torch.logspace(-2, 2, n_frequencies)
        self.freqs = nn.Parameter(init_freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [...] -> [..., 2*n_frequencies]"""
        angles = 2 * math.pi * t.unsqueeze(-1) * self.freqs  # [..., F]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [..., 2F]


class LaplaceNeuralOperator(nn.Module):
    """
    Learnable Laplace-domain kernel operator (Appendix A.2.2, Eqs. 2-3):

        kappa_hat(omega) = sum_k A_k / (omega - lambda_k)
        kappa(t) = sum_k A_k * exp(lambda_k * t),  t >= 0  (causal)

    and the resulting kernel-integral encoder (Eq. 1, discretized as Eq. 4):

        z(t) ~= sum_i kappa(t - t_i) x_i * Delta_t_i + b(t)

    Args:
        d_x: input feature dimension.
        d_z: latent dimension. ASSUMED=64 per Figure 3's caption hint
            (SIR confidence 0.35 -- not confirmed as the paper's actual value).
        n_poles: number of Laplace-domain poles K. ASSUMED, unspecified in paper.
        n_fourier: number of Fourier frequencies for the bias function. ASSUMED.
    """

    def __init__(self, d_x: int, d_z: int = 64, n_poles: int = 16, n_fourier: int = 8) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.n_poles = n_poles

        # Poles lambda_k: parameterized as -softplus(raw) + i*imag to guarantee
        # Re(lambda_k) < 0 (stability), per "learnable poles ... with Re(lambda_k)<0".
        self.pole_real_raw = nn.Parameter(torch.randn(n_poles) * 0.5)
        self.pole_imag = nn.Parameter(torch.randn(n_poles) * 0.5)

        # Residue matrices A_k: real and imaginary parts, [K, d_z, d_x] each.
        self.A_real = nn.Parameter(torch.randn(n_poles, d_z, d_x) * 0.02)
        self.A_imag = nn.Parameter(torch.randn(n_poles, d_z, d_x) * 0.02)

        self.time_embed = _TimeEmbedding(n_fourier)
        self.bias_mlp = nn.Sequential(
            nn.Linear(2 * n_fourier, d_z),
            nn.Tanh(),
            nn.Linear(d_z, d_z),
        )

    def poles(self) -> torch.Tensor:
        """lambda_k = -softplus(pole_real_raw) + i*pole_imag, guarantees Re(lambda_k)<0."""
        real = -torch.nn.functional.softplus(self.pole_real_raw)
        return torch.complex(real, self.pole_imag)  # [K] complex

    def kernel(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        kappa(delta_t) = sum_k A_k * exp(lambda_k * delta_t), causal (delta_t >= 0
        assumed by caller; values for delta_t < 0 are zeroed to enforce causality).

        Args:
            delta_t: [*] tensor of time differences (t - t_i).
        Returns:
            [*, d_z, d_x] real-valued kernel evaluations.
        """
        lam = self.poles()  # [K] complex
        A = torch.complex(self.A_real, self.A_imag)  # [K, d_z, d_x]
        exponent = lam.reshape(*([1] * delta_t.dim()), -1) * delta_t.unsqueeze(-1)
        decay = torch.exp(exponent)  # [*, K] complex
        kern = torch.einsum("...k,kzx->...zx", decay, A).real
        causal_mask = (delta_t >= 0).float().unsqueeze(-1).unsqueeze(-1)
        return kern * causal_mask

    def forward(self, x: torch.Tensor, t: torch.Tensor, eval_times: torch.Tensor) -> torch.Tensor:
        """
        Encode irregular observations into a latent function evaluated at
        `eval_times` (Eq. 4, left-Riemann-sum discretization).

        Args:
            x: [B, N, d_x] observed feature vectors.
            t: [B, N] observation timestamps (assumed sorted ascending).
            eval_times: [M] fixed grid of times at which to evaluate z(t),
                shared across the batch (feeds the SDE solver downstream).
        Returns:
            z: [B, M, d_z] latent function values.
        """
        B, N, _ = x.shape
        M = eval_times.shape[0]

        delta_t = eval_times.view(1, M, 1) - t.view(B, 1, N)  # [B, M, N]
        kern = self.kernel(delta_t)  # [B, M, N, d_z, d_x]

        dt_i = torch.diff(t, dim=1, prepend=t[:, :1])  # [B, N]
        weighted_x = x * dt_i.unsqueeze(-1)  # [B, N, d_x]
        z = torch.einsum("bjizx,bix->bjz", kern, weighted_x)  # [B, M, d_z]

        bias = self.bias_mlp(self.time_embed(eval_times))  # [M, d_z]
        return z + bias.unsqueeze(0)
