"""
gmlp/models/toeplitz.py
-----------------------
Toeplitz-constrained spatial linear layer for gMLP.

Paper Section 4 + Appendix C: "Pay Attention to MLPs" (arXiv:2105.08050)

In MLM experiments the spatial weight matrix W ∈ R^{n×n} is constrained
to be a Toeplitz matrix: W_ij = w_{i-j} for a single learnable vector
w ∈ R^{2n-1}. This exploits the fact that gMLPs organically learn
Toeplitz-like W on MLM tasks (shift invariance), while halving redundant
parameters. The constraint is empirically quality-neutral (Appendix C).

For vision tasks (not shift-invariant), W is unconstrained [n,n].

Eq. 6 (SIR): W_ij = w_{i-j}, w ∈ R^{2n-1}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ToeplitzLinear(nn.Module):
    """
    Applies a spatial linear transformation W @ z + b where W is optionally
    constrained to a Toeplitz matrix (shift-invariant).

    When use_toeplitz=True  : W ∈ R^{n×n} parameterized by w ∈ R^{2n-1}
                               (Appendix C, paper Eq. 6 in SIR)
    When use_toeplitz=False : W ∈ R^{n×n} is a free learnable parameter
                               (used for vision where shift-invariance is not assumed)

    The operation acts on the *spatial* (sequence) dimension, not the channel
    dimension. The same W is shared across all channels (Section 2.1).

    Args:
        seq_len:        Sequence length n. Determines W shape.
        use_toeplitz:   If True, constrain W to Toeplitz matrix.
        w_init_std:     Std of near-zero W initialization.
                        # ASSUMED: 0.002 — paper says "near-zero" without specifying
                        # (SIR ambiguity_002, confidence=0.65) TODO:verify
        bias_init_val:  Initial value of bias b. Paper: init b=1 so f(Z)≈1 at start.

    Paper ref: Section 2.1, Eq. 2 & 6; Appendix C
    """

    def __init__(
        self,
        seq_len: int,
        use_toeplitz: bool = True,
        w_init_std: float = 0.002,   # ASSUMED — SIR ambiguity_002
        bias_init_val: float = 1.0,  # paper: init bias=1 (explicit)
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.use_toeplitz = use_toeplitz
        self.w_init_std = w_init_std

        if use_toeplitz:
            # Parameterise as single vector w ∈ R^{2n-1} (Appendix C)
            # W_ij = w[i - j + (n-1)]
            self.weight = nn.Parameter(torch.zeros(2 * seq_len - 1))
            nn.init.normal_(self.weight, mean=0.0, std=w_init_std)
        else:
            # Unconstrained W ∈ R^{n×n} for vision
            self.weight = nn.Parameter(torch.zeros(seq_len, seq_len))
            nn.init.normal_(self.weight, mean=0.0, std=w_init_std)

        # Bias b ∈ R^{n}, initialised to 1 so f_{W,b}(Z) ≈ 1 at start of training
        # This ensures each gMLP block behaves like a regular FFN initially (Section 2.1)
        self.bias = nn.Parameter(torch.ones(seq_len))

    def _build_toeplitz(self) -> Tensor:
        """
        Construct n×n Toeplitz matrix from weight vector w ∈ R^{2n-1}.
        Equivalent to the TF implementation in Appendix C, translated to PyTorch.

        The k-th diagonal (offset k from main diagonal) gets value w[k + (n-1)].

        Returns:
            W: Tensor of shape [n, n]
        """
        n = self.seq_len
        # Build Toeplitz via circulant-style index construction
        # Indices: row i, col j → weight index = i - j + (n-1)
        idx = torch.arange(n, device=self.weight.device)
        row_idx = idx.unsqueeze(1)   # [n, 1]
        col_idx = idx.unsqueeze(0)   # [1, n]
        toeplitz_idx = row_idx - col_idx + (n - 1)   # [n, n], values in [0, 2n-2]
        return self.weight[toeplitz_idx]              # [n, n]

    def get_weight_matrix(self) -> Tensor:
        """
        Returns the full [n, n] spatial weight matrix W.
        Useful for visualisation (replicating Figure 4 in paper).
        """
        if self.use_toeplitz:
            return self._build_toeplitz()
        return self.weight

    def forward(self, z: Tensor) -> Tensor:
        """
        Apply spatial projection: output[b, :, c] = W @ z[b, :, c] + b
        The same W is applied to every channel independently.

        Implements Eq. 2 (SIR): f_{W,b}(Z) = WZ + b

        Args:
            z: Tensor of shape [B, n, e] where e = d_ffn/2

        Returns:
            Tensor of shape [B, n, e]
        """
        assert z.dim() == 3, f"[ToeplitzLinear] Expected [B, n, e], got {z.shape}"
        assert z.shape[1] == self.seq_len, (
            f"[ToeplitzLinear] seq_len mismatch: expected {self.seq_len}, got {z.shape[1]}"
        )

        W = self.get_weight_matrix()           # [n, n]
        # Spatial matmul: [B, n, e] → transpose → [B, e, n] @ W.T → [B, e, n] → transpose
        # Equivalently: einsum('nm, bme -> bne', W, z)
        out = torch.einsum("nm,bme->bne", W, z)   # [B, n, e]
        out = out + self.bias.unsqueeze(0).unsqueeze(-1)  # broadcast [1, n, 1]
        return out

    def __repr__(self) -> str:
        return (
            f"ToeplitzLinear(seq_len={self.seq_len}, "
            f"use_toeplitz={self.use_toeplitz}, "
            f"w_init_std={self.w_init_std})"
        )
