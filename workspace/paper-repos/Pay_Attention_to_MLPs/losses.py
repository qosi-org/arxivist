"""
gmlp/training/losses.py
-----------------------
Loss functions and LR schedulers for gMLP training.

Paper: "Pay Attention to MLPs" (arXiv:2105.08050)

Losses:
  - MLMLoss:             Cross-entropy on masked tokens only (NLP pretraining)
  - ClassificationLoss:  Cross-entropy with optional label smoothing (vision / GLUE)
  - QALoss:              Sum of start + end span cross-entropy (SQuAD)

LR Schedulers (Appendix A.1, A.2):
  - LinearWarmupDecay:   Warmup then linear decay to 0 (NLP default)
  - CosineWarmupDecay:   Warmup then cosine decay (vision default)

Paper ref: Appendix A.1 (vision), Appendix A.2 (NLP), Tables 7–9
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class MLMLoss(nn.Module):
    """
    Masked Language Modelling cross-entropy loss.
    Ignores positions where labels=-100 (non-masked / padding tokens).

    Paper: standard BERT MLM objective (referenced Section 4).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            logits: [B, n, vocab_size]
            labels: [B, n] — original token ids at masked positions, -100 elsewhere

        Returns:
            Scalar cross-entropy loss over masked positions.
        """
        return F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )


class ClassificationLoss(nn.Module):
    """
    Cross-entropy classification loss with optional label smoothing.
    Used for: SST-2, MNLI (NLP finetuning) and ImageNet (vision training).

    Paper Table 7: label_smoothing=0.1 for vision.
    Paper Table 9: no label smoothing mentioned for NLP finetuning (default 0).
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            logits: [B, num_classes] — raw (pre-softmax) logits
            labels: [B] int64 (hard labels) or [B, num_classes] float (soft, Mixup/CutMix)

        Returns:
            Scalar loss.
        """
        if labels.dim() == 2:
            # Soft labels from Mixup/CutMix — manual cross-entropy
            log_prob = F.log_softmax(logits, dim=-1)
            loss = -(labels * log_prob).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        return loss


class QALoss(nn.Module):
    """
    Span extraction QA loss: sum of cross-entropy over start and end positions.
    Used for SQuAD v1.1 and v2.0 finetuning.

    Paper Table 6: gMLPlarge achieves 89.5 F1 on SQuAD v1.1 without self-attention.
    aMLPlarge achieves 92.2 / 85.4 F1 on v1.1/v2.0 (outperforms BERTlarge).
    """

    def forward(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        start_positions: Tensor,
        end_positions: Tensor,
    ) -> Tensor:
        """
        Args:
            start_logits:    [B, n] — logits for start token position
            end_logits:      [B, n] — logits for end token position
            start_positions: [B] — gold start positions
            end_positions:   [B] — gold end positions

        Returns:
            Scalar: (loss_start + loss_end) / 2
        """
        seq_len = start_logits.size(1)
        start_positions = start_positions.clamp(0, seq_len - 1)
        end_positions = end_positions.clamp(0, seq_len - 1)
        loss_start = F.cross_entropy(start_logits, start_positions)
        loss_end = F.cross_entropy(end_logits, end_positions)
        return (loss_start + loss_end) / 2


# ---------------------------------------------------------------------------
# LR Schedulers
# ---------------------------------------------------------------------------

def get_linear_warmup_decay(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Linear warmup from 0 to peak lr, then linear decay to 0.
    Paper Appendix A.2: used for all NLP (pretraining + finetuning) experiments.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_warmup_decay(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Linear warmup then cosine decay.
    Paper Appendix A.1: used for ImageNet vision training.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(name: str, optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    if name == "linear":
        return get_linear_warmup_decay(optimizer, warmup_steps, total_steps)
    elif name == "cosine":
        return get_cosine_warmup_decay(optimizer, warmup_steps, total_steps)
    elif name == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError(f"Unknown lr_schedule: '{name}'. Choose 'linear', 'cosine', or 'constant'.")
