"""
Modality projection heads for cross-modal alignment.

Paper reference: Section 3.3 Eq. 7 (h^(m) = p_m(z^(m))); Appendix H,
"Cross-modal alignment" — two-layer MLP, hidden width 64, output dim 64.
SIR module: cross_modal_projection_heads (confidence 0.65 — the projection
head architecture itself is explicit at 0.8, but the modality-pair scope is
ambiguous; see SIR ambiguities[1] and losses.py::nce_alignment_loss).
"""

from __future__ import annotations

import torch
from torch import nn


class ModalityProjectionHead(nn.Module):
    """Two-layer MLP projecting a modality-specific latent branch for alignment.

    Args:
        hidden_dim: input embedding dimension (128).
        proj_hidden_width: hidden width of the 2-layer MLP (64).
        output_dim: output projection dimension d_c (64).
    """

    def __init__(self, hidden_dim: int = 128, proj_hidden_width: int = 64, output_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, proj_hidden_width),
            nn.ReLU(),
            nn.Linear(proj_hidden_width, output_dim),
        )

    def forward(self, z_m: torch.Tensor) -> torch.Tensor:
        """Project a modality-specific embedding branch.

        Args:
            z_m: [N, hidden_dim] modality-specific latent representation.

        Returns:
            h: [N, output_dim] projection used in the NCE alignment loss.
        """
        assert z_m.dim() == 2 and z_m.shape[1] == self.hidden_dim, (
            f"Expected [N, {self.hidden_dim}], got {tuple(z_m.shape)}"
        )
        return self.net(z_m)

    def __repr__(self) -> str:  # noqa: D105
        return f"ModalityProjectionHead(hidden_dim={self.hidden_dim}, output_dim={self.output_dim})"
