"""
models/symbolic_bottleneck.py
================================
Module 3: differentiable symbolic bottleneck (Section 4.3, Appendix A.6).
Distils latent dynamics into a sparse, human-readable combination of basis
functions applied to the RAW input features (not the latent state -- see
Appendix A.6.1: "a neural module that outputs a weighted combination of
basis functions computed from the raw input features").

IMPORTANT GAP (documented in architecture_plan.json risk assessment): the
paper never enumerates its actual basis-function library, nor shows any
example distilled formula anywhere, despite this being one of the paper's
four headline contributions. BasisLibrary.compute_features below implements
a reasonable, clearly-documented default (moving averages, ratios,
differences, rolling variance at a handful of lags) and export_formula()
directly addresses this gap by printing the actual distilled expression.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BasisLibrary:
    """
    Computes a library of elementary basis functions over a raw input
    window (Appendix A.6.1: moving averages, ratios, differences,
    variances "and other elementary operations").

    ASSUMED lag set [2,5,10,20] -- not enumerated in the paper.
    """

    def __init__(self, n_channels: int, lags: List[int] | None = None) -> None:
        self.n_channels = n_channels
        self.lags = lags or [2, 5, 10, 20]

    @property
    def n_basis(self) -> int:
        # Per channel: len(lags) moving averages + len(lags) differences +
        # len(lags) rolling variances + 1 ratio (last/first value).
        return self.n_channels * (3 * len(self.lags) + 1)

    def feature_names(self) -> List[str]:
        names = []
        for c in range(self.n_channels):
            for lag in self.lags:
                names.append(f"ch{c}_ma{lag}")
            for lag in self.lags:
                names.append(f"ch{c}_diff{lag}")
            for lag in self.lags:
                names.append(f"ch{c}_var{lag}")
            names.append(f"ch{c}_ratio_last_first")
        return names

    def compute_features(self, x_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_window: [B, L, C] raw input window (L timesteps, C channels).
        Returns:
            [B, n_basis] basis features.
        """
        B, L, C = x_window.shape
        assert C == self.n_channels, f"expected {self.n_channels} channels, got {C}"
        feats = []
        for c in range(C):
            xc = x_window[:, :, c]  # [B, L]
            for lag in self.lags:
                window = xc[:, -min(lag, L):]
                feats.append(window.mean(dim=1, keepdim=True))
            for lag in self.lags:
                lag = min(lag, L - 1)
                feats.append((xc[:, -1] - xc[:, -1 - lag]).unsqueeze(-1))
            for lag in self.lags:
                window = xc[:, -min(lag, L):]
                feats.append(window.var(dim=1, unbiased=False, keepdim=True))
            eps = 1e-8
            feats.append((xc[:, -1] / (xc[:, 0] + eps)).unsqueeze(-1))
        return torch.cat(feats, dim=-1)  # [B, n_basis]


class SymbolicBottleneck(nn.Module):
    """
    Sparse linear combination over the basis library (Eq. 12):
        y_hat_symb = sum_k w_k * f_k(x)
    with an L1 penalty (Eq. 13) and optional Gumbel-Softmax relaxation
    (Appendix A.6.3) over candidate basis-function selection.
    """

    def __init__(self, n_basis: int) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.w = nn.Parameter(torch.randn(n_basis) * 0.01)
        self.selection_logits = nn.Parameter(torch.zeros(n_basis))

    def forward(self, basis_features: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Args:
            basis_features: [B, n_basis].
            tau: Gumbel-Softmax temperature (annealed toward 0 during training,
                per Appendix A.6.3).
            hard: if True, use a straight-through hard selection.
        Returns:
            [B] y_hat_symb predictions.
        """
        if tau > 0:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.selection_logits) + 1e-20) + 1e-20)
            soft_mask = torch.softmax((self.selection_logits + gumbel_noise) / tau, dim=-1)
            if hard:
                hard_mask = torch.zeros_like(soft_mask)
                hard_mask[soft_mask.argmax()] = 1.0
                mask = (hard_mask - soft_mask).detach() + soft_mask
            else:
                mask = soft_mask * self.n_basis  # rescale so an "all-selected" default resembles no masking
        else:
            mask = torch.ones_like(self.selection_logits)
        return (basis_features * (self.w * mask)[None, :]).sum(-1)

    def l1_penalty(self, lambda_symb: float) -> torch.Tensor:
        """L_symb = lambda_symb * ||w||_1 (Eq. 13)."""
        return lambda_symb * self.w.abs().sum()

    def export_formula(self, feature_names: List[str], top_k: int = 10) -> str:
        """
        Print the top-|w_k| weighted terms as a human-readable formula.
        This directly addresses a documented gap in the paper: no example
        distilled formula is ever shown despite interpretability being one
        of ARTEMIS's four headline contributions.
        """
        weights = self.w.detach().cpu()
        idx = torch.argsort(weights.abs(), descending=True)[:top_k]
        terms = [f"{weights[i].item():+.4f}*{feature_names[i]}" for i in idx]
        return "y_hat_symb ~= " + " ".join(terms)
