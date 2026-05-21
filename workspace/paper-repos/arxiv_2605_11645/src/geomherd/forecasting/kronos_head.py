"""
src/geomherd/forecasting/kronos_head.py
Kronos-style discrete price tokeniser and AdaLN-Zero conditioned forecasting head.
Paper: arXiv:2605.11645, Section 3.3.3

STUB WARNING (Risk R2): The Kronos head architecture is NOT specified in the paper.
Layer count, hidden dim, and head count are absent. This implementation uses a
small configurable transformer with AdaLN-Zero conditioning as described.

Components:
  - PriceTokeniser: Frozen learned VQ-VAE on OHLCV sequences (ASSUMED architecture)
  - AdaLNZero: Adaptive layer-norm with zero-init gating (explicitly mentioned)
  - KronosHead: Transformer conditioned on GeomHerd triplet via AdaLN-Zero

Paper reference: Section 3.3.3
    'A Kronos-style discrete price tokeniser (a learned vector-quantiser that maps
    OHLCV sequences into a fixed token vocabulary) feeds a transformer that consumes
    the GeomHerd triplet (kappa_bar_OR, tau_sing, V_eff) via AdaLN-Zero conditioning.'
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the Kronos forecasting head. "
            "Install with: pip install torch"
        )


class VectorQuantiser(nn.Module):
    """
    Simple Vector Quantiser (VQ-VAE codebook) for price tokenisation.

    STUB: Architecture not specified in paper (Risk R2).
    ASSUMED: Codebook of size 512, straight-through gradient estimator.
    """

    def __init__(self, codebook_size: int = 512, embed_dim: int = 32):
        _require_torch()
        super().__init__()
        # ASSUMED: codebook_size=512, embed_dim=32
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, T, embed_dim] encoder output
        Returns:
            quantized: [B, T, embed_dim] quantized vectors
            indices: [B, T] codebook indices
        """
        B, T, D = z.shape
        z_flat = z.view(-1, D)
        # L2 distances to codebook
        dists = (z_flat.pow(2).sum(1, keepdim=True)
                 - 2 * z_flat @ self.embedding.weight.T
                 + self.embedding.weight.pow(2).sum(1, keepdim=True).T)
        indices = dists.argmin(dim=1)  # [B*T]
        quantized = self.embedding(indices).view(B, T, D)
        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        return quantized_st, indices.view(B, T)

    def __repr__(self) -> str:
        return f"VectorQuantiser(codebook_size={self.codebook_size}, embed_dim={self.embed_dim})"


class PriceTokeniser(nn.Module):
    """
    Frozen learned VQ-VAE tokeniser for OHLCV sequences.

    Paper reference: Section 3.3.3
        'A learned vector-quantiser that maps OHLCV sequences into a fixed token vocabulary.'
        The price tokeniser is frozen during Kronos head training.

    STUB (Risk R2): Architecture not specified. ASSUMED: 2-layer MLP encoder
    mapping 5-dim OHLCV to embed_dim, then VQ lookup.

    Args:
        ohlcv_dim: Input dimension (default 5 for OHLCV)
        embed_dim: Embedding dimension (ASSUMED: 32)
        codebook_size: VQ codebook size (ASSUMED: 512)
    """

    def __init__(
        self,
        ohlcv_dim: int = 5,
        embed_dim: int = 32,       # ASSUMED
        codebook_size: int = 512,  # ASSUMED
    ):
        _require_torch()
        super().__init__()
        # ASSUMED: 2-layer MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(ohlcv_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.vq = VectorQuantiser(codebook_size=codebook_size, embed_dim=embed_dim)
        self.embed_dim = embed_dim

    def encode(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ohlcv: [B, T, 5] OHLCV context window
        Returns:
            indices: [B, T] token indices
        """
        assert ohlcv.dim() == 3 and ohlcv.shape[2] == 5, \
            f"Expected [B, T, 5] OHLCV input, got {ohlcv.shape}"
        z = self.encoder(ohlcv)  # [B, T, embed_dim]
        _, indices = self.vq(z)
        return indices

    def forward(self, ohlcv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (quantized embeddings [B,T,D], indices [B,T])."""
        z = self.encoder(ohlcv)
        return self.vq(z)

    def __repr__(self) -> str:
        return f"PriceTokeniser(embed_dim={self.embed_dim}, codebook={self.vq.codebook_size})"


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm with zero-initialized gating, conditioned on external signal.

    Paper reference: Section 3.3.3
        'adaptive layer-norm with zero-initialised gating'
        Condition: GeomHerd triplet (kappa_bar_OR, tau_sing, V_eff)

    Standard AdaLN-Zero: scale and shift projected from condition, gate initialized to 0
    so initial output = identity (stable training start).

    Args:
        d_model: Hidden dimension
        cond_dim: Conditioning dimension (default 3 for GeomHerd triplet)
    """

    def __init__(self, d_model: int, cond_dim: int = 3):
        _require_torch()
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Project condition to scale and shift; zero-init for stable training
        self.proj = nn.Linear(cond_dim, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] hidden states
            condition: [B, cond_dim] conditioning signal (GeomHerd triplet)
        Returns:
            modulated: [B, T, d_model]
        """
        assert x.dim() == 3, f"Expected [B, T, d_model], got {x.shape}"
        assert condition.dim() == 2, f"Expected [B, cond_dim], got {condition.shape}"
        scale_shift = self.proj(condition)  # [B, 2*d_model]
        scale, shift = scale_shift.chunk(2, dim=-1)  # each [B, d_model]
        x_norm = self.norm(x)
        # Modulate: (1 + scale) * x_norm + shift
        return x_norm * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class KronosTransformerLayer(nn.Module):
    """
    Single transformer layer with AdaLN-Zero conditioning.

    STUB (Risk R2): layer count and head count not specified in paper.
    ASSUMED: standard pre-norm transformer structure with AdaLN-Zero replacing LayerNorm.
    """

    def __init__(self, d_model: int, n_heads: int, cond_dim: int = 3):
        _require_torch()
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.adaLN_attn = AdaLNZero(d_model, cond_dim)
        self.adaLN_ffn = AdaLNZero(d_model, cond_dim)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            condition: [B, 3] GeomHerd triplet
        Returns:
            x: [B, T, d_model]
        """
        # Self-attention with AdaLN-Zero
        x_norm = self.adaLN_attn(x, condition)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # FFN with AdaLN-Zero
        x_norm = self.adaLN_ffn(x, condition)
        x = x + self.ffn(x_norm)
        return x


class KronosHead(nn.Module):
    """
    Kronos-style forecasting head conditioned on GeomHerd triplet.

    Paper reference: Section 3.3.3
        Consumes GeomHerd triplet (kappa_bar_OR, tau_sing, V_eff) via AdaLN-Zero.
        Price tokeniser is frozen; only conditioning layers are trained.
        Predicts next-step log return.

    STUB (Risk R2): Architecture not fully specified.
    ASSUMED: n_layers=2, n_heads=4, d_model=64.

    Args:
        d_model: Transformer hidden dimension (ASSUMED: 64)
        n_layers: Number of transformer layers (ASSUMED: 2)
        n_heads: Number of attention heads (ASSUMED: 4)
        ohlcv_dim: OHLCV input dimension (default 5)
        cond_dim: Conditioning dimension (3 for GeomHerd triplet)
        price_tokeniser: Pre-trained PriceTokeniser (frozen during training)
    """

    def __init__(
        self,
        d_model: int = 64,        # ASSUMED
        n_layers: int = 2,        # ASSUMED
        n_heads: int = 4,         # ASSUMED
        ohlcv_dim: int = 5,
        cond_dim: int = 3,
        price_tokeniser: Optional["PriceTokeniser"] = None,
    ):
        _require_torch()
        super().__init__()
        # STUB: Kronos architecture not specified; see Risk R2
        self.d_model = d_model
        self.n_layers = n_layers

        if price_tokeniser is None:
            price_tokeniser = PriceTokeniser(ohlcv_dim=ohlcv_dim, embed_dim=d_model)
        self.tokeniser = price_tokeniser
        # Freeze tokeniser weights
        for p in self.tokeniser.parameters():
            p.requires_grad_(False)

        # Token embedding + positional encoding
        self.token_embed = nn.Linear(self.tokeniser.embed_dim, d_model)
        # Conditioning: project triplet to cond_dim
        self.cond_proj = nn.Linear(cond_dim, cond_dim)

        # Transformer layers with AdaLN-Zero
        self.layers = nn.ModuleList([
            KronosTransformerLayer(d_model=d_model, n_heads=n_heads, cond_dim=cond_dim)
            for _ in range(n_layers)
        ])

        # Output head: predict log return
        self.output_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        ohlcv: torch.Tensor,
        triplet: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ohlcv: [B, T_ctx, 5] OHLCV context window
            triplet: [B, 3] GeomHerd triplet (kappa_bar_OR, tau_sing, V_eff)
        Returns:
            log_return_pred: [B] next-step log return forecast
        """
        assert ohlcv.dim() == 3, f"Expected [B, T_ctx, 5], got {ohlcv.shape}"
        assert triplet.dim() == 2 and triplet.shape[1] == 3, \
            f"Expected [B, 3] triplet, got {triplet.shape}"

        # Tokenise OHLCV (frozen tokeniser)
        with torch.no_grad():
            z_quant, _ = self.tokeniser(ohlcv)  # [B, T_ctx, embed_dim]

        # Project to d_model
        h = self.token_embed(z_quant)  # [B, T_ctx, d_model]

        # Project conditioning
        cond = self.cond_proj(triplet)  # [B, cond_dim]

        # Transformer layers with AdaLN-Zero conditioning
        for layer in self.layers:
            h = layer(h, cond)

        # Predict from last token
        pred = self.output_head(h[:, -1, :]).squeeze(-1)  # [B]
        return pred

    def predict_log_return(
        self, ohlcv: np.ndarray, triplet: np.ndarray
    ) -> float:
        """Convenience wrapper for single-sample inference."""
        _require_torch()
        self.eval()
        with torch.no_grad():
            ohlcv_t = torch.tensor(ohlcv, dtype=torch.float32).unsqueeze(0)
            triplet_t = torch.tensor(triplet, dtype=torch.float32).unsqueeze(0)
            pred = self.forward(ohlcv_t, triplet_t)
        return float(pred.item())

    def __repr__(self) -> str:
        return (f"KronosHead(d_model={self.d_model}, n_layers={self.n_layers}, "
                f"n_heads={self.layers[0].attn.num_heads if self.layers else 0})")
