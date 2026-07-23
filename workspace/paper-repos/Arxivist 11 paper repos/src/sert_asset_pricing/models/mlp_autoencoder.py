"""
MLP Autoencoder — shared building block used for two distinct roles in the paper:

1. **Pre-training dimension-projection module** (Section 4.2, Fig. 6-7): projects the
   182 raw sorted-portfolio factors up to d_model=420 to align with the cross-attention
   / decoder dimension, described as "a generalized PCA process ... without the
   restriction of orthogonality and linearity" (Section 4.2). This ASSUMES a single
   hidden layer (SIR ambiguities[0], confidence 0.4 — see sir.json).

2. **In-block feedforward autoencoder** (Section 4.1.2, Appendix C): the standard
   Transformer FFN sub-layer, sized to `ffn_hidden_ratio * d_model` (0.7 for
   standard/pretrained/SERT groups, 0.2 for LNF variants per Section 5.1).

Appendix C gives the general multi-hidden-layer autoencoder recursion (Eq. C68-C74);
this implementation supports an arbitrary number of hidden layers via
`hidden_layers`, defaulting to 1 per the ambiguity-resolution assumption above.
"""
from __future__ import annotations

import torch
from torch import nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLPAutoencoder(nn.Module):
    """Single- or multi-hidden-layer MLP autoencoder (encoder + decoder MLP stack).

    Implements the generalized autoencoder recursion of Appendix C (Eq. C67-C74):
    Z = f_en(X; theta_en); Y_hat = f_de(Z; theta_de).

    Args:
        in_dim: input feature dimension.
        out_dim: output feature dimension (may differ from in_dim, e.g. 182 -> 420
            for the pre-training projection module).
        hidden_ratio: latent-layer width as a fraction of max(in_dim, out_dim).
            Paper: 0.7 for main-body FFN, 0.2 for LNF variants (Section 5.1).
        hidden_layers: number of hidden layers in the encoder MLP stack (and
            symmetrically in the decoder stack). ASSUMED=1 (SIR confidence 0.4).
        activation: activation function name, one of {"relu", "gelu", "tanh"}.
            ASSUMED="relu" — not explicitly named in the paper (confidence 0.6).
        dropout: dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_ratio: float = 0.7,
        hidden_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert in_dim > 0 and out_dim > 0, "in_dim and out_dim must be positive"
        assert 0.0 < hidden_ratio <= 1.0, f"hidden_ratio must be in (0,1], got {hidden_ratio}"
        assert hidden_layers >= 1, "hidden_layers must be >= 1"
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}', choose from {list(_ACTIVATIONS)}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        hidden_dim = max(1, round(hidden_ratio * max(in_dim, out_dim)))
        self.hidden_dim = hidden_dim
        act_cls = _ACTIVATIONS[activation]

        # Encoder: in_dim -> hidden_dim (x hidden_layers)
        enc_layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(hidden_layers):
            enc_layers += [nn.Linear(prev, hidden_dim), act_cls(), nn.Dropout(dropout)]
            prev = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: hidden_dim -> out_dim (x hidden_layers, last layer linear/no activation
        # to allow unrestricted regression output, matching Appendix C's f_de projection).
        dec_layers: list[nn.Module] = []
        prev = hidden_dim
        for i in range(hidden_layers):
            is_last = i == hidden_layers - 1
            next_dim = out_dim if is_last else hidden_dim
            dec_layers.append(nn.Linear(prev, next_dim))
            if not is_last:
                dec_layers += [act_cls(), nn.Dropout(dropout)]
            prev = next_dim
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project x through encoder -> latent -> decoder.

        Args:
            x: [..., in_dim] input tensor (any leading batch/sequence dims).

        Returns:
            [..., out_dim] projected tensor.
        """
        assert x.shape[-1] == self.in_dim, (
            f"Expected last dim {self.in_dim}, got {x.shape[-1]} (full shape {tuple(x.shape)})"
        )
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat

    def __repr__(self) -> str:
        return (
            f"MLPAutoencoder(in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )
