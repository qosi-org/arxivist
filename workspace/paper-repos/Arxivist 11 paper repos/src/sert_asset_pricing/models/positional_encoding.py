"""
Sinusoidal Positional Encoding (SPE).

Implements Eq. 1-2 of "Asset Pricing in Pre-trained Transformers" (arXiv:2505.01575),
Section 4.1.1. Positional codes are ADDED (not concatenated) to the linearly-embedded
factor inputs: X' = X + PE, as stated in the paper text following Eq. 1-2.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, as in Vaswani et al. (2017), Eq. 1-2 of the paper.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        d_model: embedding dimension (paper uses d_model=420, aligned to num_stocks).
        max_len: maximum sequence length to precompute encodings for.
    """

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        assert d_model > 0, f"d_model must be positive, got {d_model}"

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # Guard against odd d_model where cos slice is one shorter than sin slice.
        pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]

        # Registered as a buffer so it moves with .to(device) but is not a trainable param.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input embedding.

        Args:
            x: [B, T, d_model] embedded input.

        Returns:
            [B, T, d_model] = x + PE[:, :T, :]
        """
        assert x.dim() == 3, f"Expected [B,T,D], got {tuple(x.shape)}"
        seq_len = x.size(1)
        assert seq_len <= self.pe.size(1), (
            f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)} "
            "used to precompute positional encodings."
        )
        return x + self.pe[:, :seq_len, :].to(dtype=x.dtype)

    def __repr__(self) -> str:
        return f"SinusoidalPositionalEncoding(d_model={self.pe.size(-1)}, max_len={self.pe.size(1)})"
