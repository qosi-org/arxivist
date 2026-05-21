"""Iteration-based step LR schedule with optional warmup (paper Sec. 4.2).

Paper rules:
  - Base LR 0.1, dropped by 10x at iterations 32000 and 48000, total 64000 iterations.
  - ResNet-110: warm-up at LR 0.01 for the first 400 iterations, then switch to base LR.
"""
from __future__ import annotations

from typing import Iterable

import torch


class StepLRWithWarmup:
    """Manual iteration-based scheduler.

    The PyTorch built-in schedulers operate per-epoch and don't match the paper's iteration-based
    drops cleanly, so we drive the LR directly.

    Args:
        optimizer: torch optimizer (LR will be written into all param groups).
        base_lr: post-warmup LR (paper: 0.1).
        warmup_lr: warmup LR (paper: 0.01 for ResNet-110).
        warmup_iterations: number of iterations to hold warmup_lr before stepping to base_lr.
            Set to 0 to disable warmup.
        drop_iterations: iterations at which to divide LR by `drop_factor` (paper: [32000, 48000]).
        drop_factor: LR division factor (paper: 10.0).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_lr: float,
        warmup_iterations: int,
        drop_iterations: Iterable[int],
        drop_factor: float = 10.0,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_lr = float(warmup_lr)
        self.warmup_iterations = int(warmup_iterations)
        self.drop_iterations = sorted(int(x) for x in drop_iterations)
        self.drop_factor = float(drop_factor)
        if self.drop_factor <= 0:
            raise ValueError(f"drop_factor must be > 0, got {drop_factor}")

    def lr_for_iteration(self, iteration: int) -> float:
        if self.warmup_iterations > 0 and iteration < self.warmup_iterations:
            return self.warmup_lr

        lr = self.base_lr
        for drop_at in self.drop_iterations:
            if iteration >= drop_at:
                lr = lr / self.drop_factor
        return lr

    def step(self, iteration: int) -> float:
        """Set optimizer LR for the given (0-indexed) iteration. Returns the LR applied."""
        lr = self.lr_for_iteration(iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
