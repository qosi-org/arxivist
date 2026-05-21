"""Loss functions.

The paper uses standard categorical cross-entropy (implicit). No label smoothing — that
practice post-dates the paper.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy. logits: [B, C], targets: [B] int64. Returns scalar mean loss."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B,C], got {tuple(logits.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"targets must be [B], got {tuple(targets.shape)}")
    return F.cross_entropy(logits, targets, reduction="mean")
