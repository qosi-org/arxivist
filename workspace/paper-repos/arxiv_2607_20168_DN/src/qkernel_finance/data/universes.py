"""
Two universe constructions -- the paper's central methodological contrast (Sec 3.2).

PointInTimeUniverse: constructible in real time, used for the main 170-window
study. StaticScreenUniverse: conditions on full-sample survival, deliberately
retained by the paper as "the treatment arm of a natural experiment in
research design" to demonstrate how it manufactures a spurious quantum
advantage (Sec 6).
"""
from __future__ import annotations

import pandas as pd


class PointInTimeUniverse:
    """Top-450 by trailing 20-day mean float market cap, 252-day price coverage > 90% (Sec 3.2)."""

    def __init__(self, pool_size: int = 450, trailing_coverage_days: int = 252, coverage_threshold: float = 0.90, mktcap_window_days: int = 20) -> None:
        self.pool_size = pool_size
        self.trailing_coverage_days = trailing_coverage_days
        self.coverage_threshold = coverage_threshold
        self.mktcap_window_days = mktcap_window_days

    def eligible_stocks(self, date: pd.Timestamp, price_panel: pd.DataFrame, mktcap_panel: pd.DataFrame) -> list[str]:
        """
        Args:
            date: the trading day t at which to determine eligibility.
            price_panel: daily price panel with columns [date, ticker, price] (or NaN for non-trading).
            mktcap_panel: daily float-market-cap panel with columns [date, ticker, float_mktcap].

        Returns:
            List of ticker symbols in the point-in-time investable pool at `date`.
            Uses no information beyond `date` (Sec 3.2: "No information beyond t enters").
        """
        window_start = date - pd.Timedelta(days=int(self.trailing_coverage_days * 1.5))  # calendar buffer for trading days
        trailing_prices = price_panel[(price_panel["date"] > window_start) & (price_panel["date"] <= date)]

        coverage = trailing_prices.groupby("ticker")["price"].apply(lambda s: s.notna().mean())
        eligible_by_coverage = coverage[coverage > self.coverage_threshold].index

        recent_mktcap = mktcap_panel[
            (mktcap_panel["date"] > date - pd.Timedelta(days=int(self.mktcap_window_days * 1.5)))
            & (mktcap_panel["date"] <= date)
        ]
        mean_mktcap = recent_mktcap.groupby("ticker")["float_mktcap"].mean()
        mean_mktcap = mean_mktcap.loc[mean_mktcap.index.intersection(eligible_by_coverage)]

        return mean_mktcap.sort_values(ascending=False).head(self.pool_size).index.tolist()

    def __repr__(self) -> str:  # noqa: D105
        return f"PointInTimeUniverse(pool_size={self.pool_size})"


class StaticScreenUniverse:
    """Fixed pool of the 450 stocks with highest full-sample (2020-2025) price coverage (Sec 3.2).

    Deliberately unimplementable in real time -- conditions on survival through
    the sample's end. Used only by the diagnostic study (Sec 6) to demonstrate
    how a common applied-literature construction manufactures a spurious result.
    """

    def __init__(self, pool_size: int = 450) -> None:
        self.pool_size = pool_size

    def eligible_stocks(self, price_panel: pd.DataFrame) -> list[str]:
        """
        Args:
            price_panel: the full evaluation-period price panel (all dates),
                columns [date, ticker, price].

        Returns:
            Fixed list of ticker symbols, same for every date in the study.
        """
        coverage = price_panel.groupby("ticker")["price"].apply(lambda s: s.notna().mean())
        return coverage.sort_values(ascending=False).head(self.pool_size).index.tolist()

    def __repr__(self) -> str:  # noqa: D105
        return f"StaticScreenUniverse(pool_size={self.pool_size})"
