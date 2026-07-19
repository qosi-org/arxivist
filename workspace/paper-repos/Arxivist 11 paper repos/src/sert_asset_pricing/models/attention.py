"""
Masked multi-head self-attention and cross-attention.

Implements Eq. 3-11 of "Asset Pricing in Pre-trained Transformers" (arXiv:2505.01575),
Section 4.1.3:

- Eq. 3:   Q = W^Q X,  K = W^K X,  V = W^V X
- Eq. 4:   Attention Score = QK^T / sqrt(d_model)
- Eq. 5-6: causal mask M_ij = 0 if i<j else -inf   (prevents future data leakage,
           mandatory for time-series Transformers per Section 4.1.3)
- Eq. 7:   Masked Attention(Q,K,V) = softmax(QK^T/sqrt(d_model) + M) V
- Eq. 8-9: multi-head variant, concat + linear W^O
- Eq. 10-11: cross-attention, Q from decoder, K/V from encoder
"""
from __future__ import annotations

import torch
from torch import nn


class CausalMask:
    """Builds the additive causal mask of Eq. 5-6: 0 for i<j, -inf for i>j."""

    @staticmethod
    def build(seq_len: int, device: torch.device) -> torch.Tensor:
        """Construct a [T, T] additive causal mask.

        Args:
            seq_len: sequence length T.
            device: torch device to place the mask on.

        Returns:
            [T, T] float tensor with 0 on/below diagonal (allowed) and -inf above (masked),
            matching Eq. 6: M_ij = 0 if i<j else -inf, applied so position i cannot attend
            to future positions j > i.
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)  # zero on/below diagonal, -inf strictly above
        return mask


class MultiHeadSelfAttention(nn.Module):
    """Masked multi-head self-attention (Eq. 3-9).

    Args:
        d_model: model/embedding dimension.
        num_heads: number of attention heads h. Must divide d_model.
        dropout: dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Eq. 3
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Eq. 9
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked multi-head self-attention.

        Args:
            x: [B, T, d_model] input.
            mask: [T, T] additive mask (from CausalMask.build).

        Returns:
            [B, T, d_model] attention output.
        """
        assert x.dim() == 3, f"Expected [B,T,D], got {tuple(x.shape)}"
        b, t, d = x.shape
        assert d == self.d_model, f"Expected last dim {self.d_model}, got {d}"

        q = self.w_q(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)  # [B,h,T,d_k]
        k = self.w_k(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)

        # Eq. 4 + Eq. 8: scaled dot product per head, plus causal mask.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B,h,T,T]
        scores = scores + mask  # broadcast [T,T] -> [B,h,T,T]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,h,T,d_k]
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)  # Concat heads, Eq. 9
        return self.w_o(out)

    def __repr__(self) -> str:
        return f"MultiHeadSelfAttention(d_model={self.d_model}, num_heads={self.num_heads})"


class MultiHeadCrossAttention(nn.Module):
    """Masked multi-head cross-attention (Eq. 10-11).

    Q is derived from the decoder output; K,V are derived from the encoder output.
    Only used in the full Transformer variants (PretrainedTransformer, PretrainedTransformerLNF,
    StandardTransformer benchmark); SERT/EncoderOnlyTransformer never instantiate this module.

    Args:
        d_model: model/embedding dimension (paper's d_H, encoder/decoder hidden dim).
        num_heads: number of attention heads.
        dropout: dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Eq. 10: Q_de = H_de W^Q
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # K_en = H_en W^K
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # V_en = H_en W^V
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q_in: torch.Tensor, kv_in: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply masked multi-head cross-attention.

        Args:
            q_in: [B, T_dec, d_model] decoder hidden states (source of Q).
            kv_in: [B, T_enc, d_model] encoder hidden states (source of K, V).
            mask: [T_dec, T_enc] additive cross-attention mask (Eq. 11's M_cross).

        Returns:
            [B, T_dec, d_model] cross-attention output.
        """
        assert q_in.dim() == 3 and kv_in.dim() == 3, "q_in and kv_in must be [B,T,D]"
        b, t_dec, d = q_in.shape
        _, t_enc, _ = kv_in.shape
        assert d == self.d_model, f"Expected last dim {self.d_model}, got {d}"

        q = self.w_q(q_in).view(b, t_dec, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(kv_in).view(b, t_enc, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(kv_in).view(b, t_enc, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # Eq. 11
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t_dec, self.d_model)
        return self.w_o(out)

    def __repr__(self) -> str:
        return f"MultiHeadCrossAttention(d_model={self.d_model}, num_heads={self.num_heads})"
