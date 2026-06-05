"""
data/dataset.py — Dataset and data loading utilities for CRSP unbalanced panel.

Handles the unbalanced panel structure: ~10,000 stocks per month, but each stock
has a different number of observations. Missing values are masked.

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Section V.A.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
import numpy as np


class AssetPricingDataset(Dataset):
    """
    Dataset for the unbalanced panel of stock returns and characteristics.

    Wraps three tensors:
        macro_series: [T, 178] macroeconomic time series
        firm_chars:   [T, N, 46] firm characteristics (quantile-normalized, 0-padded for missing)
        returns:      [T, N] excess returns (NaN for missing)
        obs_mask:     [T, N] bool mask — True where stock i is observed at time t

    Panel weights T_i/T are computed from obs_mask and used in the loss (Eq. 3).

    Args:
        macro_series: numpy array [T, 178]
        firm_chars: numpy array [T, N, 46]
        returns: numpy array [T, N]  (NaN for missing)
        device: torch device
    """

    def __init__(
        self,
        macro_series: np.ndarray,
        firm_chars: np.ndarray,
        returns: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.T = macro_series.shape[0]
        self.N = returns.shape[1]

        self.macro_series = torch.tensor(macro_series, dtype=torch.float32, device=device)
        self.firm_chars = torch.tensor(firm_chars, dtype=torch.float32, device=device)

        ret = torch.tensor(returns, dtype=torch.float32, device=device)
        self.obs_mask = ~torch.isnan(ret)                    # [T, N] bool
        self.returns = torch.nan_to_num(ret, nan=0.0)        # replace NaN with 0

        # Panel weights: T_i/T per stock
        T_i = self.obs_mask.float().sum(dim=0)               # [N]
        self.panel_weights = T_i / self.T                    # [N]

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all data tensors (full panel — used directly in training)."""
        return self.macro_series, self.firm_chars, self.returns, self.panel_weights

    def __len__(self) -> int:
        return self.T

    def __repr__(self) -> str:
        obs = self.obs_mask.float().sum().item()
        return (
            f"AssetPricingDataset(T={self.T}, N={self.N}, "
            f"obs={int(obs)}, fill_rate={obs/(self.T*self.N):.2%})"
        )


def make_synthetic_dataset(
    T: int = 600,
    N: int = 500,
    num_macro: int = 178,
    num_chars: int = 46,
    fill_rate: float = 0.85,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> AssetPricingDataset:
    """
    Generate synthetic data for unit testing and smoke tests.

    Matches simulation setup from Section IV:
        T=600, N=500, SDF_SR≈1.0, sigma_e=1.0

    Args:
        T: time periods (paper: 600)
        N: number of stocks (paper: 500)
        fill_rate: fraction of (t,i) pairs that are observed

    Returns:
        AssetPricingDataset with synthetic data
    """
    rng = np.random.default_rng(seed)

    # Macro: random walk (non-stationary, like real macro data)
    macro = np.cumsum(rng.standard_normal((T, num_macro)) * 0.1, axis=0)

    # Firm chars: uniform quantile [0, 1], cross-sectionally normalized
    chars = rng.uniform(0, 1, (T, N, num_chars)).astype(np.float32)
    chars = (chars - chars.mean(axis=1, keepdims=True))  # center per month

    # Returns: simple one-factor model (Section IV simulation)
    F = rng.standard_normal(T) * 0.316 + 0.1  # SR≈1, sigma_F=0.316
    beta = rng.standard_normal((T, N))
    idio = rng.standard_normal((T, N))
    returns = beta * F[:, None] + idio
    returns = returns.astype(np.float32)

    # Introduce missing values
    mask = rng.uniform(0, 1, (T, N)) < (1 - fill_rate)
    returns[mask] = np.nan

    return AssetPricingDataset(macro, chars, returns, device=device)
