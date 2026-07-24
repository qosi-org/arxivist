"""
Training objective (Sec 4.3): "minimising the Huber loss, which is robust
to the right-skewed distribution of SPR values."

WARNING: low-confidence implementation (SIR mathematical_spec[2], confidence
0.55). The paper names the Huber loss but never states its delta (transition
threshold between quadratic and linear regions). We default to delta=1.0
(PyTorch's `nn.HuberLoss` default) -- see implementation_assumptions[3] in
sir.json. SPR distances in this paper range roughly 0-2000, so delta may
need tuning; it is fully exposed via config (training.huber_delta).
"""
from __future__ import annotations

import torch
from torch import nn


class HuberRegressionLoss(nn.Module):
    """Thin, explicit wrapper around torch.nn.HuberLoss for SPR-distance regression.

    Args:
        delta: Huber loss transition threshold. ASSUMED default 1.0 (not
            stated in the paper) -- tune via config.training.huber_delta.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta
        self._loss_fn = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(self, pred: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            pred: [B] predicted SPR distances.
            target: [B] ground-truth (phangorn heuristic) SPR distances.

        Returns:
            Scalar mean Huber loss.
        """
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        return self._loss_fn(pred, target)

    def __repr__(self) -> str:  # noqa: D105
        return f"HuberRegressionLoss(delta={self.delta})"
