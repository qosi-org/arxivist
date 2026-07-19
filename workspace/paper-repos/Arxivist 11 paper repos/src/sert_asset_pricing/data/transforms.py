"""
Factor preprocessing utilities.

Implements Section 3's data-cleaning rule: exclude any factor with more than
`missing_value_threshold` (40%) missing values, to "maintain a reasonable data
quality" before the MLP-autoencoder pre-training module handles remaining gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FactorPreprocessor:
    """Drops factor columns with excessive missingness (Section 3).

    Args:
        missing_value_threshold: drop any column whose fraction of missing values
            exceeds this threshold (paper uses 0.4).
    """

    missing_value_threshold: float = 0.4
    dropped_columns_: list[str] | None = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit missingness stats on `df` and return the filtered dataframe.

        Args:
            df: raw factor dataframe, columns = factor names, index = dates.

        Returns:
            Filtered dataframe with high-missingness columns dropped.
        """
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        missing_frac = df.isna().mean(axis=0)
        keep_cols = missing_frac[missing_frac <= self.missing_value_threshold].index.tolist()
        self.dropped_columns_ = [c for c in df.columns if c not in keep_cols]
        return df[keep_cols].copy()

    def __repr__(self) -> str:
        n_dropped = len(self.dropped_columns_) if self.dropped_columns_ is not None else "unfit"
        return f"FactorPreprocessor(missing_value_threshold={self.missing_value_threshold}, dropped={n_dropped})"
