"""Loss functions.

- classification_loss: cross-entropy for downstream tasks.
- causal_lm_loss: next-token prediction loss (SIR mathematical_spec, conf 0.9),
  used only in the reference from-scratch pretraining path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy over class logits. logits [B, C], labels [B]."""
    return F.cross_entropy(logits, labels)


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 4) -> torch.Tensor:
    """Next-token prediction loss (HyenaDNA pretraining objective).

    Eq: L = -sum_t log p(x_t | x_<t). logits [B, L, V], targets [B, L].
    Shifted so position t predicts token t+1. ignore_index masks padding.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=ignore_index,
    )
