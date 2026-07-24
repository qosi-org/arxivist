"""
Walk-forward windowing engine (Sec 4.1): trailing 252-day training windows,
20-day rebalancing, cross-sections sampled every 3rd day within the window.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd


class WalkForwardEngine:
    """Generates (train_window, rebalance_date) pairs for the main or diagnostic study."""

    def __init__(self, training_window_days: int = 252, rebalance_frequency_days: int = 20, cross_section_sample_every_n_days: int = 3) -> None:
        self.training_window_days = training_window_days
        self.rebalance_frequency_days = rebalance_frequency_days
        self.cross_section_sample_every_n_days = cross_section_sample_every_n_days

    def iter_windows(self, panel_df: pd.DataFrame, num_windows: int | None = None) -> Iterator[dict]:
        """
        Args:
            panel_df: full daily panel with a `date` column, sorted ascending.
            num_windows: if set, caps the number of windows yielded (for
                smoke tests); otherwise yields as many as the panel supports.

        Yields:
            dict with keys: `rebalance_date`, `train_window` (DataFrame, every
            3rd trading day within the trailing 252-day window, label realized
            20 days before the rebalance date so every label is available),
            `all_dates` (sorted unique trading dates in the panel).
        """
        all_dates = sorted(panel_df["date"].unique())
        n = self.training_window_days
        step = self.rebalance_frequency_days
        sample_every = self.cross_section_sample_every_n_days

        count = 0
        # First possible rebalance date needs at least n+step prior trading days
        # (n for the training window, step so every label is realized -- Sec 4.1).
        start_idx = n + step
        for idx in range(start_idx, len(all_dates), step):
            if num_windows is not None and count >= num_windows:
                break
            rebalance_date = all_dates[idx]
            window_end_idx = idx - step  # training window ends 20 days before rebalance date
            window_start_idx = window_end_idx - n
            if window_start_idx < 0:
                continue
            window_dates = all_dates[window_start_idx:window_end_idx][::sample_every]
            train_window = panel_df[panel_df["date"].isin(window_dates)]

            yield {
                "rebalance_date": rebalance_date,
                "train_window": train_window,
                "all_dates": all_dates,
            }
            count += 1

    def __repr__(self) -> str:  # noqa: D105
        return f"WalkForwardEngine(window={self.training_window_days}d, rebalance_every={self.rebalance_frequency_days}d)"
