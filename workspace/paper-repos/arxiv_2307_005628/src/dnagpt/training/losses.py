"""DNAGPT pre-training losses (Sec 3.2, Eq 1).

    L = lambda * MSE_loss + CrossEntropy_loss     (lambda = 0.01)

The cross-entropy covers the next-token-prediction and sequence-order tasks
(both over discrete tokens); the MSE covers the GC-content regression (and any
other number tokens). This module provides the combined loss and helpers to
build the two DNA-specific pre-training targets.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

LAMBDA = 0.01  # paper Sec 3.2


def combined_pretrain_loss(
    class_logits: torch.Tensor,
    class_targets: torch.Tensor,
    reg_values: torch.Tensor,
    reg_targets: torch.Tensor,
    lambda_: float = LAMBDA,
    ignore_index: int = -100,
) -> torch.Tensor:
    """L = lambda * MSE + CrossEntropy (Eq 1).

    Args:
        class_logits: [B, T, V] next-token / order logits.
        class_targets: [B, T] token ids (ignore_index where not scored).
        reg_values: [B, K, 1] predicted numbers (GC content etc.).
        reg_targets: [B, K] target numbers.
    """
    ce = F.cross_entropy(
        class_logits.reshape(-1, class_logits.size(-1)),
        class_targets.reshape(-1),
        ignore_index=ignore_index,
    )
    if reg_targets.numel() > 0:
        mse = F.mse_loss(reg_values.reshape(reg_targets.shape), reg_targets)
    else:
        mse = torch.zeros((), device=class_logits.device)
    return lambda_ * mse + ce


def gc_content_targets(seqs: List[str]) -> torch.Tensor:
    """GC ratio in [0,1] per sequence (target for the GC regression task)."""
    out = []
    for s in seqs:
        s = s.upper()
        out.append((s.count("G") + s.count("C")) / max(len(s), 1))
    return torch.tensor(out, dtype=torch.float32)
