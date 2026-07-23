"""
Feedforward reconstruction decoder.

Paper reference: Section 3.3 Eq. 5 (X_hat = g_phi(Z)); Appendix H, "Decoder
architecture" — one hidden layer of width 2*hidden_dim followed by a linear
output layer. SIR module: reconstruction_decoder (confidence 0.9).
"""

from __future__ import annotations

import torch
from torch import nn


class ReconstructionDecoder(nn.Module):
    """Maps spot embeddings Z back to the full concatenated feature space.

    Args:
        hidden_dim: embedding dimension d (128 in the paper).
        output_dim: total concatenated feature dimension D = sum_b d_b.
        decoder_hidden_width: width of the single hidden layer (256 = 2*128).
    """

    def __init__(
        self, hidden_dim: int = 128, output_dim: int = 0, decoder_hidden_width: int = 256
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(
                "output_dim must be set to the total concatenated feature "
                "dimension D (sum of all modality block dims)."
            )
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_width),
            nn.ReLU(),
            nn.Linear(decoder_hidden_width, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the full feature matrix from embeddings.

        Args:
            z: [N, hidden_dim] embeddings.

        Returns:
            x_hat: [N, output_dim] reconstructed features.
        """
        assert z.dim() == 2 and z.shape[1] == self.hidden_dim, (
            f"Expected [N, {self.hidden_dim}], got {tuple(z.shape)}"
        )
        return self.net(z)

    def __repr__(self) -> str:  # noqa: D105
        return f"ReconstructionDecoder(hidden_dim={self.hidden_dim}, output_dim={self.output_dim})"
