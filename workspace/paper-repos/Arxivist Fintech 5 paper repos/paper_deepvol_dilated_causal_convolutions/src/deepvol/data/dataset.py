"""
PyTorch Dataset for DeepVol.
Handles multi-asset NASDAQ-100 intraday high-frequency data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class VolatilityDataset(Dataset):
    """
    Dataset of (intraday_returns_sequence, realised_variance_target) pairs.

    Expects preprocessed numpy arrays:
      X: [N, 1, T*J]  float32  — input intraday return sequences
      y: [N, 1]        float32  — realised variance targets

    Args:
        X: Input array [N, 1, T*J]
        y: Target array [N, 1]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @classmethod
    def from_numpy_files(cls, x_path: str, y_path: str) -> "VolatilityDataset":
        X = np.load(x_path)
        y = np.load(y_path)
        return cls(X, y)

    def __repr__(self):
        return f"VolatilityDataset(n={len(self)}, seq_len={self.X.shape[-1]})"


class SyntheticVolatilityDataset(Dataset):
    """
    Synthetic dataset for testing and notebook demos.
    Generates random returns with GARCH-like volatility clustering.
    No downloads required.
    """

    def __init__(self, n_samples: int = 500, seq_len: int = 78, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Simulate simple AR(1) variance process for synthetic data
        vol = 0.01
        returns, rvs = [], []
        for _ in range(n_samples + 1):
            vol = 0.1 * rng.standard_normal() ** 2 + 0.9 * vol
            day_returns = rng.normal(0, np.sqrt(vol), size=seq_len).astype(np.float32)
            returns.append(day_returns)
            rvs.append(float(np.sum(day_returns ** 2)))

        X = np.array(returns[:-1])[:, np.newaxis, :]   # [N, 1, J]
        y = np.array(rvs[1:], dtype=np.float32)[:, np.newaxis]  # [N, 1]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __repr__(self):
        return f"SyntheticVolatilityDataset(n={len(self)}, seq_len={self.X.shape[-1]})"
