"""
Per-window factor rotation and top-8 selection (Sec 4.1).

Active set: characteristics with |window rank IC| >= 0.015 (minimum 6 kept),
sign-corrected. Top-8 set: the 8 largest |IC| characteristics, shared
verbatim by all top-8 models (classical and quantum) -- the paper's main
kernel-swap control input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class FactorRotationSelector:
    """Computes window-level rank IC per characteristic and selects the active/top-8 sets."""

    def __init__(self, active_set_ic_threshold: float = 0.015, active_set_min_size: int = 6, top8_size: int = 8) -> None:
        self.active_set_ic_threshold = active_set_ic_threshold
        self.active_set_min_size = active_set_min_size
        self.top8_size = top8_size

    def _window_ic(self, window_panel: pd.DataFrame, characteristic_cols: list[str], return_col: str) -> pd.Series:
        """Average cross-sectional rank IC per characteristic over the training window's dates."""
        ics: dict[str, list[float]] = {c: [] for c in characteristic_cols}
        for _date, day_df in window_panel.groupby("date"):
            y = day_df[return_col].values
            for c in characteristic_cols:
                x = day_df[c].values
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                ic, _p = spearmanr(x, y)
                if not np.isnan(ic):
                    ics[c].append(ic)
        return pd.Series({c: (np.mean(v) if v else 0.0) for c, v in ics.items()})

    def select(self, window_panel: pd.DataFrame, characteristic_cols: list[str], return_col: str = "fwd_return_20d") -> tuple[list[str], list[str]]:
        """
        Args:
            window_panel: the trailing 252-day training window's panel (sampled
                every 3rd day per Sec 4.1), with `date`, `characteristic_cols`,
                and `return_col` columns.
            characteristic_cols: candidate characteristic column names.
            return_col: the target column (20-day forward return).

        Returns:
            (active_set, top8) -- both lists of column names. `top8` always has
            exactly `top8_size` entries (assuming enough characteristics are
            supplied); `active_set` has at least `active_set_min_size` entries.
        """
        ic = self._window_ic(window_panel, characteristic_cols, return_col)
        abs_ic = ic.abs().sort_values(ascending=False)

        active_mask = abs_ic >= self.active_set_ic_threshold
        active_set = abs_ic[active_mask].index.tolist()
        if len(active_set) < self.active_set_min_size:
            active_set = abs_ic.index[: self.active_set_min_size].tolist()

        top8 = abs_ic.index[: self.top8_size].tolist()

        # Sign-correction: flip characteristics with negative IC so higher
        # values consistently mean "higher expected return" for all downstream models.
        self.last_signs_ = {c: (1.0 if ic[c] >= 0 else -1.0) for c in set(active_set) | set(top8)}

        return active_set, top8

    def __repr__(self) -> str:  # noqa: D105
        return f"FactorRotationSelector(ic_threshold={self.active_set_ic_threshold}, top8_size={self.top8_size})"
