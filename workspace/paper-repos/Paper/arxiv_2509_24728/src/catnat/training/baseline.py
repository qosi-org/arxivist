"""
Baseline estimators for REINFORCE variance reduction.

Implements the Leave-One-Out (LOO) multi-sample baseline used in the GSL experiment.
The baseline reduces gradient variance without introducing bias.

Paper: Appendix E.3 — "multi-sample baseline where, for each sample in a batch
of M sampled graphs, the baseline b is constructed using the estimate of the
loss from the other M-1 samples."

Kool et al. (2019) — "Buy 4 REINFORCE samples, get a baseline for free."
"""

import torch
from torch import Tensor


class LOOBaseline:
    """Leave-One-Out (LOO) baseline for REINFORCE.

    For M graph samples, the baseline for sample m is the mean loss
    of all OTHER M-1 samples:
        b_m = (1/(M-1)) * Σ_{n≠m} L(A_n)

    This is a control variate that reduces gradient variance without bias.

    Paper: Appendix E.3. Confidence: 0.99 (explicitly stated).
    """

    def compute(self, losses: Tensor) -> Tensor:
        """Compute LOO baselines for a batch of M-sample losses.

        Args:
            losses: Per-sample losses, shape [M, B] (M samples, B batch elements).

        Returns:
            Baseline per sample, shape [M, B].
        """
        assert losses.dim() == 2, (
            f"LOOBaseline expects losses of shape [M, B], got {losses.shape}"
        )
        M, B = losses.shape
        if M < 2:
            raise ValueError(f"LOO baseline requires M >= 2 samples, got M={M}.")

        # Sum of all losses per batch element, then subtract own contribution
        total = losses.sum(dim=0, keepdim=True)      # [1, B]
        loo_sum = total - losses                       # [M, B]  (leave m out)
        baseline = loo_sum / (M - 1)                  # [M, B]
        return baseline.detach()                       # detach: baseline is not differentiated


class MovingAverageBaseline:
    """Exponential moving average baseline for single-sample REINFORCE.

    Not used in the paper's main experiments (LOO is preferred for M=32),
    but included as a lightweight alternative for M=1 cases.

    Args:
        alpha: EMA decay rate. Default 0.99.
    """

    def __init__(self, alpha: float = 0.99) -> None:
        self.alpha = alpha
        self._value: float = 0.0
        self._initialized: bool = False

    def update(self, loss: float) -> float:
        """Update the baseline with the current loss and return the new value.

        Args:
            loss: Current batch mean loss.

        Returns:
            Current baseline value.
        """
        if not self._initialized:
            self._value = loss
            self._initialized = True
        else:
            self._value = self.alpha * self._value + (1.0 - self.alpha) * loss
        return self._value

    @property
    def value(self) -> float:
        """Current baseline value."""
        return self._value

    def __repr__(self) -> str:
        return f"MovingAverageBaseline(alpha={self.alpha}, value={self._value:.4f})"
