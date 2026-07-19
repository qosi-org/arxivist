"""
Encoder / decoder blocks.

Implements Eq. 12-13 (Section 4.1.4) combined with the attention (Eq. 3-11) and
feedforward autoencoder (Section 4.1.2) sub-layers:

- Eq. 12 (post-LN, standard):   X' = LayerNorm(X + Sublayer(X))
- Eq. 13 (pre-LN, LNF variant): X' = X + Sublayer(LayerNorm(X))

Figure 1 (standard Transformer) shows one encoder block containing:
  Multi-head self-attention -> Add&Norm -> FFN autoencoder -> Add&Norm
and one decoder block containing:
  Masked self-attention -> Add&Norm -> Cross-attention -> Add&Norm -> FFN autoencoder -> Add&Norm

N* (number of stacked blocks) = 1 for all models examined in this paper (Section 4).
"""
from __future__ import annotations

import torch
from torch import nn

from sert_asset_pricing.models.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from sert_asset_pricing.models.mlp_autoencoder import MLPAutoencoder


class _AddNorm(nn.Module):
    """Wraps a sublayer with either post-LN (Eq. 12) or pre-LN / LNF (Eq. 13) residual connection."""

    def __init__(self, d_model: int, layer_norm_first: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer_norm_first = layer_norm_first

    def forward(self, x: torch.Tensor, sublayer_out_fn) -> torch.Tensor:
        if self.layer_norm_first:
            # Eq. 13: X' = X + sublayer(LayerNorm(X))
            return x + sublayer_out_fn(self.norm(x))
        # Eq. 12: X' = LayerNorm(X + sublayer(X))
        return self.norm(x + sublayer_out_fn(x))


class EncoderBlock(nn.Module):
    """One Transformer encoder block: masked self-attention + FFN autoencoder, each with Add&Norm.

    Args:
        d_model: model dimension.
        num_heads: number of self-attention heads.
        ffn_hidden_ratio: FFN autoencoder hidden width as a fraction of d_model
            (0.7 standard, 0.2 for LNF variants — Section 5.1).
        layer_norm_first: if True, use pre-LN (Eq. 13, LNF variant); else post-LN (Eq. 12).
        activation: FFN activation function name.
        dropout: dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden_ratio: float = 0.7,
        layer_norm_first: bool = False,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.ffn = MLPAutoencoder(
            in_dim=d_model,
            out_dim=d_model,
            hidden_ratio=ffn_hidden_ratio,
            hidden_layers=1,
            activation=activation,
            dropout=dropout,
        )
        self.attn_add_norm = _AddNorm(d_model, layer_norm_first)
        self.ffn_add_norm = _AddNorm(d_model, layer_norm_first)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run one encoder block forward pass.

        Args:
            x: [B, T, d_model] input.
            mask: [T, T] causal mask.

        Returns:
            [B, T, d_model] encoder block output.
        """
        assert x.dim() == 3, f"Expected [B,T,D], got {tuple(x.shape)}"
        x = self.attn_add_norm(x, lambda h: self.self_attn(h, mask))
        x = self.ffn_add_norm(x, self.ffn)
        return x


class DecoderBlock(nn.Module):
    """One Transformer decoder block: masked self-attn + cross-attn + FFN autoencoder,
    each with Add&Norm. Only instantiated by the full Transformer variants
    (PretrainedTransformer, PretrainedTransformerLNF, StandardTransformer benchmark).

    Args:
        d_model: model dimension.
        num_heads: number of attention heads (used for both self- and cross-attention).
        ffn_hidden_ratio: FFN autoencoder hidden width fraction.
        layer_norm_first: pre-LN (True) vs post-LN (False).
        activation: FFN activation function name.
        dropout: dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden_ratio: float = 0.7,
        layer_norm_first: bool = False,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout=dropout)
        self.ffn = MLPAutoencoder(
            in_dim=d_model,
            out_dim=d_model,
            hidden_ratio=ffn_hidden_ratio,
            hidden_layers=1,
            activation=activation,
            dropout=dropout,
        )
        self.self_attn_add_norm = _AddNorm(d_model, layer_norm_first)
        self.cross_attn_add_norm = _AddNorm(d_model, layer_norm_first)
        self.ffn_add_norm = _AddNorm(d_model, layer_norm_first)

    def forward(
        self,
        y: torch.Tensor,
        enc_out: torch.Tensor,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder block forward pass.

        Args:
            y: [B, T_dec, d_model] decoder input (shifted-right output embedding + PE).
            enc_out: [B, T_enc, d_model] encoder output (source of cross-attn K,V).
            self_mask: [T_dec, T_dec] causal mask for decoder self-attention.
            cross_mask: [T_dec, T_enc] mask for cross-attention (Eq. 11's M_cross).

        Returns:
            [B, T_dec, d_model] decoder block output.
        """
        assert y.dim() == 3 and enc_out.dim() == 3, "y and enc_out must be [B,T,D]"
        y = self.self_attn_add_norm(y, lambda h: self.self_attn(h, self_mask))
        y = self.cross_attn_add_norm(y, lambda h: self.cross_attn(h, enc_out, cross_mask))
        y = self.ffn_add_norm(y, self.ffn)
        return y
