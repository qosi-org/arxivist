"""GRPO group-relative advantage normalization — Eq. 5 in the paper.

    A_i = (r_i - mean({r_j})) / std({r_j})

Shared by both the vanilla-GRPO baseline and DS-GRPO — this part is standard GRPO, not the
paper's novel contribution (that's in differential_smoothing.py).
"""
from __future__ import annotations

import torch


class GRPOAdvantage:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def compute(self, group_rewards: torch.Tensor) -> torch.Tensor:
        """group_rewards: [B, G] -> advantages: [B, G], normalized within each group of G."""
        assert group_rewards.dim() == 2, f"expected [B, G], got {group_rewards.shape}"
        mean = group_rewards.mean(dim=1, keepdim=True)
        std = group_rewards.std(dim=1, keepdim=True)
        return (group_rewards - mean) / (std + self.eps)
