"""
Top-level model classes for the six model families examined in
"Asset Pricing in Pre-trained Transformers" (arXiv:2505.01575), Section 4 and Table 2.

| Class                       | Config family name           | Decoder | Cross-attn | Pretrain MLP-AE | LNF   |
|------------------------------|------------------------------|---------|------------|------------------|-------|
| PretrainedTransformer         | pretrained_transformer        | Yes     | Yes        | Yes              | No    |
| PretrainedTransformerLNF       | pretrained_transformer_lnf    | Yes     | Yes        | Yes              | Yes   |
| StandardTransformer (bench)   | standard_transformer          | Yes     | Yes        | No               | No    |
| SERT                          | sert                          | No      | No         | Yes              | No    |
| SERTLNF                        | sert_lnf                      | No      | No         | Yes              | Yes   |
| EncoderOnlyTransformer (bench)| encoder_only_transformer      | No      | No         | No               | No    |

Model naming follows the paper's Appendix D mapping (P_Trans_H*, SERT_H*, C_Trans_H*, En_Trans_H*).
"""
from __future__ import annotations

import torch
from torch import nn

from sert_asset_pricing.models.attention import CausalMask
from sert_asset_pricing.models.blocks import DecoderBlock, EncoderBlock
from sert_asset_pricing.models.mlp_autoencoder import MLPAutoencoder
from sert_asset_pricing.models.positional_encoding import SinusoidalPositionalEncoding


class _BaseTransformer(nn.Module):
    """Shared scaffolding: input embedding, optional pretrain-MLP autoencoder, positional encoding."""

    def __init__(
        self,
        input_factor_dim: int,
        d_model: int,
        use_pretrain_module: bool,
        pretrain_hidden_layers: int,
        activation: str,
        dropout: float,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        self.use_pretrain_module = use_pretrain_module
        self.input_factor_dim = input_factor_dim
        self.d_model = d_model

        if use_pretrain_module:
            # Section 4.2: MLP autoencoder projects 182 raw factors -> d_model (420), a
            # generalized non-linear PCA. Present only in *_pretrained_* / sert* families.
            self.pretrain_autoencoder = MLPAutoencoder(
                in_dim=input_factor_dim,
                out_dim=d_model,
                hidden_ratio=0.7,
                hidden_layers=pretrain_hidden_layers,
                activation=activation,
                dropout=dropout,
            )
            self.input_embedding = nn.Identity()
        else:
            # Benchmark models (Group 3): no pretraining module, direct linear embedding
            # from raw factor_dim -> d_model.
            self.pretrain_autoencoder = None
            self.input_embedding = nn.Linear(input_factor_dim, d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)

    def _embed_input(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Project raw factors to d_model and add positional encoding.

        Args:
            x_raw: [B, T, input_factor_dim] raw sorted-portfolio factors.

        Returns:
            [B, T, d_model] embedded + positionally-encoded input.
        """
        assert x_raw.shape[-1] == self.input_factor_dim, (
            f"Expected last dim {self.input_factor_dim}, got {x_raw.shape[-1]}"
        )
        if self.use_pretrain_module:
            x = self.pretrain_autoencoder(x_raw)  # [B,T,182] -> [B,T,420]
        else:
            x = self.input_embedding(x_raw)  # [B,T,182] -> [B,T,d_model]
        return self.pos_encoding(x)


class PretrainedTransformer(_BaseTransformer):
    """Full encoder-decoder Transformer with MLP-autoencoder pre-training module (Fig. 6).

    Corresponds to paper models P_Trans_H1..H7 (Group 2). Uses post-LN (Add & LayerNorm).
    """

    def __init__(
        self,
        input_factor_dim: int = 182,
        d_model: int = 420,
        num_heads: int = 3,
        num_blocks: int = 1,
        ffn_hidden_ratio: float = 0.7,
        pretrain_hidden_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.1,
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__(
            input_factor_dim, d_model, use_pretrain_module=True,
            pretrain_hidden_layers=pretrain_hidden_layers, activation=activation, dropout=dropout,
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ffn_hidden_ratio, layer_norm_first, activation, dropout)
            for _ in range(num_blocks)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ffn_hidden_ratio, layer_norm_first, activation, dropout)
            for _ in range(num_blocks)
        ])
        self.output_embedding = nn.Linear(1, d_model)  # embeds shifted-right scalar returns
        self.output_dense = nn.Linear(d_model, 1)  # Section 4.4: linear dense layer, not softmax

    def forward(self, x_raw: torch.Tensor, y_shifted: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pre-trained encoder-decoder Transformer.

        Args:
            x_raw: [B, T, 182] raw factor inputs.
            y_shifted: [B, T, 1] shifted-right target returns (teacher-forcing input to decoder).

        Returns:
            [B, T, 1] predicted excess returns.
        """
        device = x_raw.device
        enc_x = self._embed_input(x_raw)  # [B,T,d_model]
        enc_mask = CausalMask.build(enc_x.size(1), device)
        for block in self.encoder_blocks:
            enc_x = block(enc_x, enc_mask)

        dec_y = self.output_embedding(y_shifted)  # [B,T,d_model]
        dec_y = self.pos_encoding(dec_y)
        dec_self_mask = CausalMask.build(dec_y.size(1), device)
        cross_mask = torch.zeros(dec_y.size(1), enc_x.size(1), device=device)  # no cross-masking by default
        for block in self.decoder_blocks:
            dec_y = block(dec_y, enc_x, dec_self_mask, cross_mask)

        return self.output_dense(dec_y)

    def __repr__(self) -> str:
        return f"PretrainedTransformer(d_model={self.d_model}, blocks={len(self.encoder_blocks)})"


class StandardTransformer(_BaseTransformer):
    """Standard (non-pretrained) full Transformer benchmark. Corresponds to C_Trans_H1/H2/H4."""

    def __init__(
        self,
        input_factor_dim: int = 182,
        d_model: int = 420,
        num_heads: int = 1,
        num_blocks: int = 1,
        ffn_hidden_ratio: float = 0.7,
        activation: str = "relu",
        dropout: float = 0.1,
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__(
            input_factor_dim, d_model, use_pretrain_module=False,
            pretrain_hidden_layers=1, activation=activation, dropout=dropout,
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ffn_hidden_ratio, layer_norm_first, activation, dropout)
            for _ in range(num_blocks)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ffn_hidden_ratio, layer_norm_first, activation, dropout)
            for _ in range(num_blocks)
        ])
        self.output_embedding = nn.Linear(1, d_model)
        self.output_dense = nn.Linear(d_model, 1)

    def forward(self, x_raw: torch.Tensor, y_shifted: torch.Tensor) -> torch.Tensor:
        """See PretrainedTransformer.forward — identical shapes, no pretraining module."""
        device = x_raw.device
        enc_x = self._embed_input(x_raw)
        enc_mask = CausalMask.build(enc_x.size(1), device)
        for block in self.encoder_blocks:
            enc_x = block(enc_x, enc_mask)

        dec_y = self.output_embedding(y_shifted)
        dec_y = self.pos_encoding(dec_y)
        dec_self_mask = CausalMask.build(dec_y.size(1), device)
        cross_mask = torch.zeros(dec_y.size(1), enc_x.size(1), device=device)
        for block in self.decoder_blocks:
            dec_y = block(dec_y, enc_x, dec_self_mask, cross_mask)

        return self.output_dense(dec_y)

    def __repr__(self) -> str:
        return f"StandardTransformer(d_model={self.d_model}, blocks={len(self.encoder_blocks)})"


class SERT(_BaseTransformer):
    """Single-directional Encoder Representations from Transformer (Section 4.3, Fig. 8 left).

    Encoder-only: causal-masked self-attention + MLP-autoencoder pretraining, no decoder /
    cross-attention. Encoder FFN autoencoder output connects directly (linearly) to the
    output dense layer. Corresponds to paper models SERT_H1..H7.
    """

    def __init__(
        self,
        input_factor_dim: int = 182,
        d_model: int = 420,
        num_heads: int = 4,
        num_blocks: int = 1,
        ffn_hidden_ratio: float = 0.7,
        pretrain_hidden_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.1,
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__(
            input_factor_dim, d_model, use_pretrain_module=True,
            pretrain_hidden_layers=pretrain_hidden_layers, activation=activation, dropout=dropout,
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ffn_hidden_ratio, layer_norm_first, activation, dropout)
            for _ in range(num_blocks)
        ])
        self.output_dense = nn.Linear(d_model, 1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Forward pass through SERT (encoder-only, no teacher-forcing input needed).

        Args:
            x_raw: [B, T, 182] raw factor inputs.

        Returns:
            [B, T, 1] predicted excess returns.
        """
        device = x_raw.device
        enc_x = self._embed_input(x_raw)
        enc_mask = CausalMask.build(enc_x.size(1), device)
        for block in self.encoder_blocks:
            enc_x = block(enc_x, enc_mask)
        return self.output_dense(enc_x)

    def __repr__(self) -> str:
        return f"SERT(d_model={self.d_model}, blocks={len(self.encoder_blocks)})"


class EncoderOnlyTransformer(_BaseTransformer):
    """Standard (non-pretrained) encoder-only Transformer benchmark (Fig. 8 right).

    Corresponds to paper models En_Trans_H1/H2/H4 (Cong et al. 2021-style benchmark).
    """

    def __init__(
        self,
        input_factor_dim: int = 182,
        d_model: int = 420,
        num_heads: int = 1,
        num_blocks: int = 1,
        ffn_hidden_ratio: float = 0.7,
        activation: str = "relu",
        dropout: float = 0.1,
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__(
            input_factor_dim, d_model, use_pretrain_module=False,
            pretrain_hidden_layers=1, activation=activation, dropout=dropout,
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ffn_hidden_ratio, layer_norm_first, activation, dropout)
            for _ in range(num_blocks)
        ])
        self.output_dense = nn.Linear(d_model, 1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """See SERT.forward — identical shapes, no pretraining module."""
        device = x_raw.device
        enc_x = self._embed_input(x_raw)
        enc_mask = CausalMask.build(enc_x.size(1), device)
        for block in self.encoder_blocks:
            enc_x = block(enc_x, enc_mask)
        return self.output_dense(enc_x)

    def __repr__(self) -> str:
        return f"EncoderOnlyTransformer(d_model={self.d_model}, blocks={len(self.encoder_blocks)})"


def build_model(config: dict) -> nn.Module:
    """Factory: instantiate the correct model class from a parsed config dict.

    Args:
        config: parsed config.yaml dict (from ConfigLoader.load()).

    Returns:
        An instantiated nn.Module matching config["model"]["family"].

    Raises:
        ValueError: if config["model"]["family"] is not one of the six known families.
    """
    m = config["model"]
    family = m["family"]
    common_pretrained = dict(
        input_factor_dim=m["input_factor_dim"],
        d_model=m["d_model"],
        num_heads=m["num_heads"],
        num_blocks=m["num_blocks"],
        ffn_hidden_ratio=m["ffn_hidden_ratio_lnf"] if m.get("layer_norm_first") else m["ffn_hidden_ratio"],
        pretrain_hidden_layers=m.get("pretrain_hidden_layers", 1),
        activation=m.get("activation", "relu"),
        dropout=m.get("dropout", 0.1),
        layer_norm_first=m.get("layer_norm_first", False),
    )
    common_standard = {k: v for k, v in common_pretrained.items() if k != "pretrain_hidden_layers"}

    if family in ("pretrained_transformer", "pretrained_transformer_lnf"):
        return PretrainedTransformer(**common_pretrained)
    if family == "standard_transformer":
        return StandardTransformer(**common_standard)
    if family in ("sert", "sert_lnf"):
        return SERT(**common_pretrained)
    if family == "encoder_only_transformer":
        return EncoderOnlyTransformer(**common_standard)

    raise ValueError(f"Unknown model.family: {family!r}")
