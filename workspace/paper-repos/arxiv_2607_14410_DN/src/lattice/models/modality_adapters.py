"""
Per-modality input adapters and modality-aware fusion.

Paper reference: Section 3.3 ("Graph encoder and self-supervised objective")
and Appendix H ("Encoder architecture"). SIR modules:
`modality_input_adapters`, `modality_aware_fusion` (confidence 0.8 / 0.7).

Each of the B=5 modality blocks X^(b) in R^{N x d_b} is first projected into
a shared hidden space of size `hidden_dim` via a modality-specific linear
adapter, then fused via modality-aware mean pooling.
"""

from __future__ import annotations

import torch
from torch import nn


class ModalityInputAdapter(nn.Module):
    """Linear adapter projecting one modality block into the shared hidden space.

    Paper reference: Appendix H, "Each modality block is first projected into
    a shared hidden space using a modality-specific linear adapter."
    (SIR architecture.modules: modality_input_adapters, confidence 0.8 — the
    existence and hidden dim are explicit; whether an activation follows the
    linear layer is not stated, so we keep it a bare linear layer to avoid
    inventing an unstated nonlinearity.)

    Args:
        in_dim: input feature dimension d_b for this modality block.
        hidden_dim: shared hidden dimension (128 in the paper).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a single modality block.

        Args:
            x: [N, in_dim] float tensor.

        Returns:
            [N, hidden_dim] float tensor.
        """
        assert x.dim() == 2, f"Expected [N, in_dim], got {tuple(x.shape)}"
        assert x.shape[1] == self.in_dim, (
            f"ModalityInputAdapter configured for in_dim={self.in_dim}, "
            f"got input with last dim {x.shape[1]}"
        )
        return self.linear(x)

    def __repr__(self) -> str:  # noqa: D105
        return f"ModalityInputAdapter(in_dim={self.in_dim}, hidden_dim={self.hidden_dim})"


class ModalityAwareFusion(nn.Module):
    """Fuses B modality-adapted representations via presence-weighted mean pooling.

    Paper reference: Appendix H, "The projected modality representations are
    fused by modality-aware mean pooling (weighted by observed modality
    presence per spot)." (SIR confidence 0.7 — existence and general
    description are explicit, but the *exact* weighting formula is not
    spelled out. SIR ambiguities[3]: we ASSUME zero-imputation for absent
    modalities plus presence-mask-weighted averaging, which is the simplest
    reading consistent with the text.)

    Args:
        num_modalities: number of modality blocks B (5 in the paper).
    """

    def __init__(self, num_modalities: int = 5) -> None:
        super().__init__()
        self.num_modalities = num_modalities

    def forward(
        self, modality_reps: list[torch.Tensor], presence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Fuse modality-adapted representations.

        Args:
            modality_reps: list of B tensors, each [N, hidden_dim] — the
                output of one ModalityInputAdapter per modality block.
            presence_mask: [N, B] binary tensor; presence_mask[i, b] == 1 if
                modality b has real (non-imputed) signal at spot i, else 0.
                ASSUMPTION (SIR ambiguities[3], confidence 0.5): absent
                modalities are represented by zero rows upstream and excluded
                here via this mask rather than contributing to the mean.

        Returns:
            [N, hidden_dim] fused representation.
        """
        assert len(modality_reps) == self.num_modalities, (
            f"Expected {self.num_modalities} modality representations, "
            f"got {len(modality_reps)}"
        )
        stacked = torch.stack(modality_reps, dim=1)  # [N, B, hidden_dim]
        assert presence_mask.shape[:2] == stacked.shape[:2], (
            f"presence_mask shape {tuple(presence_mask.shape)} does not match "
            f"[N, B]={tuple(stacked.shape[:2])}"
        )
        mask = presence_mask.unsqueeze(-1).to(stacked.dtype)  # [N, B, 1]
        weighted_sum = (stacked * mask).sum(dim=1)  # [N, hidden_dim]
        denom = mask.sum(dim=1).clamp(min=1.0)  # [N, 1] avoid div-by-zero
        return weighted_sum / denom

    def __repr__(self) -> str:  # noqa: D105
        return f"ModalityAwareFusion(num_modalities={self.num_modalities})"
