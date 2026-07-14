"""
Rolling-window panel dataset over sorted-portfolio factors and stock excess returns.

Implements Section 3's windowing scheme:
- training window = 102 months
- validation window = 30 months (30% of training)
- step size = 12 months (parameters re-estimated every 12 months)
- in-sample total = 102 + 30 + 45*12 = 672 observations (Jan 1957 - Dec 2012)

Each dataset item is one rolling window: X = [T, num_factors] factor slice,
y = [T, num_stocks] excess-return slice for the same dates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class RollingWindowSpec:
    """Rolling-window configuration (Section 3)."""

    train_window: int = 102
    val_window: int = 30
    step_size: int = 12


class RollingFactorDataset(Dataset):
    """Rolling-window dataset pairing sorted-portfolio factors with stock excess returns.

    Args:
        factors: [num_dates, num_factors] dataframe, index = monthly dates.
        returns: [num_dates, num_stocks] dataframe, aligned index with `factors`.
        spec: RollingWindowSpec controlling window sizes and step.
        split: one of {"train", "val"} — selects the train or validation slice of
            each rolling window.
    """

    def __init__(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        spec: RollingWindowSpec | None = None,
        split: str = "train",
    ) -> None:
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        assert len(factors) == len(returns), "factors and returns must have equal length"
        assert (factors.index == returns.index).all(), "factors and returns must share the same date index"

        self.factors = factors
        self.returns = returns
        self.spec = spec or RollingWindowSpec()
        self.split = split
        self._window_starts = self._compute_window_starts()

    def _compute_window_starts(self) -> list[int]:
        """Compute the starting index of each rolling window."""
        n = len(self.factors)
        block = self.spec.train_window + self.spec.val_window
        starts = []
        start = 0
        while start + block <= n:
            starts.append(start)
            start += self.spec.step_size
        return starts

    def __len__(self) -> int:
        return len(self._window_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (X, y) tensors for the train or val slice of rolling window `idx`.

        Returns:
            X: [T, num_factors] float32 tensor.
            y: [T, num_stocks] float32 tensor.
            where T = train_window if split=="train" else val_window.
        """
        start = self._window_starts[idx]
        if self.split == "train":
            lo, hi = start, start + self.spec.train_window
        else:
            lo, hi = start + self.spec.train_window, start + self.spec.train_window + self.spec.val_window

        x_slice = self.factors.iloc[lo:hi].to_numpy(dtype=np.float32)
        y_slice = self.returns.iloc[lo:hi].to_numpy(dtype=np.float32)
        return torch.from_numpy(x_slice), torch.from_numpy(y_slice)

    def __repr__(self) -> str:
        return (
            f"RollingFactorDataset(n_windows={len(self)}, split={self.split!r}, "
            f"train_window={self.spec.train_window}, val_window={self.spec.val_window})"
        )
