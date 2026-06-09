"""
data/lob_dataset.py — Limit Order Book dataset loader and episode manager.

Handles loading LOBSTER message data and reconstructing LOB state history
for use in the RL environment. LOBSTER data is proprietary (see data/README_data.md).
A synthetic fallback is provided for testing.

Paper: arXiv:2301.08688 — Section 3.1, Section 5.
Data source: LOBSTER (Huang & Polak, 2011) [9]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class LOBDataset:
    """Manages LOBSTER LOB message data for episode replay.

    Each episode corresponds to one trading day's opening hour (09:30–10:30).
    The dataset loads pre-processed snapshots at 100ms resolution.

    Args:
        data_dir: Path to directory containing LOBSTER CSV files.
        asset: Ticker symbol (default 'AAPL').
        split: 'train' or 'test'.
        history_len: Number of past LOB snapshots per observation (default 100).
        action_freq_sec: Time between agent decisions in seconds (default 0.1).
    """

    # Expected columns in each LOB snapshot CSV:
    # time, bid_price, bid_volume, ask_price, ask_volume, mid_price
    COLUMNS = ["time", "bid_price", "bid_volume", "ask_price", "ask_volume", "mid_price"]

    def __init__(
        self,
        data_dir: str | Path,
        asset: str = "AAPL",
        split: str = "train",
        history_len: int = 100,
        action_freq_sec: float = 0.1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.asset = asset
        self.split = split
        self.history_len = history_len
        self.action_freq_sec = action_freq_sec

        self._episodes: list[NDArray] = []
        self._loaded = False

    def load(self) -> None:
        """Load all episodes from disk. Falls back to synthetic data if unavailable."""
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            print(
                f"[LOBDataset] Data directory '{split_dir}' not found. "
                "Generating SYNTHETIC data for testing. "
                "See data/README_data.md for real LOBSTER data setup."
            )
            self._episodes = self._generate_synthetic_episodes()
        else:
            self._episodes = self._load_from_disk(split_dir)

        self._loaded = True
        print(f"[LOBDataset] Loaded {len(self._episodes)} episodes ({self.split}).")

    def _load_from_disk(self, split_dir: Path) -> list[NDArray]:
        """Load LOBSTER episode CSVs from disk."""
        import csv
        episodes = []
        csv_files = sorted(split_dir.glob(f"{self.asset}_*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files matching '{self.asset}_*.csv' found in {split_dir}. "
                "See data/README_data.md."
            )
        for fpath in csv_files:
            rows = []
            with open(fpath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append([float(row[c]) for c in self.COLUMNS])
            episodes.append(np.array(rows, dtype=np.float32))
        return episodes

    def _generate_synthetic_episodes(
        self, num_episodes: int = 40, steps_per_episode: int = 36000
    ) -> list[NDArray]:
        """Generate synthetic LOB data for unit testing and notebook demos.

        Produces a random walk mid-price with synthetic bid/ask spread.
        steps_per_episode = 3600s / 0.1s = 36000 steps per hour.
        """
        episodes = []
        rng = np.random.default_rng(42)
        for _ in range(num_episodes):
            # Random walk mid-price around 570 (AAPL ~2012)
            log_returns = rng.normal(0, 5e-5, size=steps_per_episode)
            mid_prices = 570.0 * np.exp(np.cumsum(log_returns))

            spread = 0.01  # 1 cent spread (typical for AAPL)
            bid_prices = mid_prices - spread / 2
            ask_prices = mid_prices + spread / 2
            bid_volumes = rng.integers(100, 1000, size=steps_per_episode).astype(float)
            ask_volumes = rng.integers(100, 1000, size=steps_per_episode).astype(float)
            times = np.arange(steps_per_episode) * self.action_freq_sec

            episode = np.stack(
                [times, bid_prices, bid_volumes, ask_prices, ask_volumes, mid_prices],
                axis=1,
            ).astype(np.float32)
            episodes.append(episode)
        return episodes

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> NDArray:
        assert self._loaded, "Call .load() before accessing episodes."
        return self._episodes[idx]

    def get_mid_prices(self, episode_idx: int) -> NDArray:
        """Return mid-price array for a given episode."""
        return self[episode_idx][:, 5]  # column index 5 = mid_price

    def __repr__(self) -> str:
        return (
            f"LOBDataset(asset={self.asset}, split={self.split}, "
            f"episodes={len(self._episodes)}, loaded={self._loaded})"
        )
