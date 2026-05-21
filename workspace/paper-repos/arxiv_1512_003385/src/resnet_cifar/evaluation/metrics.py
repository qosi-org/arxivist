"""Top-1 accuracy meter."""
from __future__ import annotations

import torch


class AccuracyMeter:
    """Running top-1 accuracy / error. Reports values as percentages."""

    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.dim() != 2:
            raise ValueError(f"logits must be [B,C], got {tuple(logits.shape)}")
        if labels.dim() != 1:
            raise ValueError(f"labels must be [B], got {tuple(labels.shape)}")
        preds = logits.argmax(dim=1)
        self.correct += int((preds == labels).sum().item())
        self.total += int(labels.numel())

    def compute(self) -> tuple[float, float]:
        if self.total == 0:
            return 0.0, 0.0
        top1 = 100.0 * self.correct / self.total
        return top1, 100.0 - top1

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
