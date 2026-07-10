"""Logistic-normal distribution utilities.

Implements the bijective simplex transform h / h^-1 and the variance
annealing schedule from Section 4.2 and Section 4.3 (Eq. 12) of:
Cheridito & Weiss, "Reinforcement Learning for Trade Execution with Market
and Limit Orders", arXiv:2507.06345.
"""
from __future__ import annotations

import torch


class LogisticNormalTransform:
    """Bijective map between R^K and the interior of the simplex S^K.

    Paper reference: Section 4.2, logistic transform h and its inverse h^-1.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Map x in R^K to action a in S^K (dim K+1).

        a_k = exp(x_k) / (1 + sum_l exp(x_l)),  k = 0, ..., K-1
        a_K = 1 / (1 + sum_l exp(x_l))
        """
        assert x.dim() == 2, f"Expected [B, K], got {tuple(x.shape)}"
        # Numerically stable softmax-with-implicit-zero trick.
        zeros = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        x_aug = torch.cat([x, zeros], dim=-1)  # [B, K+1], last column implicit logit=0
        a = torch.softmax(x_aug, dim=-1)  # [B, K+1]
        return a

    @staticmethod
    def inverse(a: torch.Tensor) -> torch.Tensor:
        """Map action a in S^K (dim K+1) back to x in R^K.

        x_k = log(a_k / a_K),  k = 0, ..., K-1
        """
        assert a.dim() == 2, f"Expected [B, K+1], got {tuple(a.shape)}"
        eps = 1e-8
        a_K = a[:, -1:].clamp_min(eps)
        x = torch.log(a[:, :-1].clamp_min(eps) / a_K)
        return x

    @staticmethod
    def variance_schedule(step: int, sigma_init: float, sigma_final: float, H: int) -> float:
        """Linear variance annealing schedule, Eq. 12.

        sigma_i = (sigma_final - sigma_init) * (i - 1) / (H - 1) + sigma_init
        for i = 1, ..., H (1-indexed in the paper; `step` here is 1-indexed too).
        """
        if H <= 1:
            return sigma_final
        step = max(1, min(step, H))
        return (sigma_final - sigma_init) * (step - 1) / (H - 1) + sigma_init
