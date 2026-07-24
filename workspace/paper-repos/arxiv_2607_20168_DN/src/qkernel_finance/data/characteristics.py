"""
Firm characteristic construction and cross-sectional standardization (Sec 3.1, Table 1).

Real reproduction requires China A-share exchange snapshots and quarterly
financial statements, which are not publicly bundled with this repo (see
data/README_data.md and architecture_plan.json's risk_assessment: "High"
severity, proprietary data). This module implements the *transformation*
logic (winsorization, standardization, announcement-date alignment) against
whatever raw panel is supplied -- including the synthetic generator in
data/synthetic.py, used for smoke-testing the pipeline end-to-end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_CHARACTERISTICS = [
    "book_to_price", "sales_to_price", "earnings_yield", "dividend_yield",  # Value (4)
    "market_beta", "current_ratio",  # partial Risk & leverage (market-observable)
    "mom_12_1", "reversal_1m", "bollinger_z", "ppo", "lottery_max",  # Technical (5)
]

FUNDAMENTAL_CHARACTERISTICS = [
    "roa", "roe", "gross_margin", "net_margin", "asset_turnover", "cfo_to_ni",
    "pct_accruals", "receivables_to_profit", "wc_to_sales",  # Profitability & quality (9)
    "sales_growth", "earnings_growth", "asset_growth", "delta_gross_margin",
    "sue", "earnings_variability",  # Growth & surprises (6)
    "debt_to_asset",  # Risk & leverage (fundamental)
]


class CharacteristicBuilder:
    """Builds and standardizes the paper's firm characteristics (Sec 3.1, Table 1)."""

    def build_market_characteristics(self, price_df: pd.DataFrame, mktcap_df: pd.DataFrame) -> pd.DataFrame:
        """Market-based characteristics are point-in-time by construction (Sec 3.1).

        Args:
            price_df: daily price panel, indexed by (date, ticker).
            mktcap_df: daily market-cap/turnover panel, indexed by (date, ticker).

        Returns:
            DataFrame of raw (pre-standardization) market-based characteristics.
        """
        raise NotImplementedError(
            "Real market-characteristic construction requires the actual China A-share "
            "price/valuation/turnover feed described in Sec 3.1 -- not bundled with this "
            "repo. See data/README_data.md and data/synthetic.py for a schema-compatible "
            "synthetic generator you can use to smoke-test the rest of the pipeline."
        )

    def build_fundamental_characteristics(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """Fundamental characteristics, merged as of their announcement dates (Sec 3.1).

        Rules (Sec 3.1): single-quarter income/cash-flow items aggregated to TTM only
        when 4 consecutive quarters span <=380 days; balance-sheet items enter at latest
        disclosed level; YoY growth requires a 330-400 day span; forward-filled <=250
        trading days.

        Args:
            financials_df: quarterly financial-statement panel.

        Returns:
            DataFrame of raw (pre-standardization) fundamental characteristics,
            announcement-date aligned.
        """
        raise NotImplementedError(
            "Real fundamental-characteristic construction requires quarterly financial "
            "statement data with announcement dates -- not bundled with this repo. "
            "See data/README_data.md and data/synthetic.py."
        )

    def standardize(self, panel_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Winsorizes at +/-3 sigma and standardizes cross-sectionally, per day, over the full market.

        Implemented via groupby().transform() rather than groupby().apply()
        deliberately: apply() with group_keys=False silently drops the
        grouping column from its output on some pandas versions (a real bug
        caught during testing -- confirmed on pandas>=3.0), which would lose
        the `date_col` column entirely. transform() cannot have this failure
        mode since it never touches non-transformed columns.

        Args:
            panel_df: raw characteristic panel with a `date_col` column and one
                column per characteristic.
            date_col: name of the date column to group by.

        Returns:
            Winsorized and standardized panel (same shape, same columns).
        """
        result = panel_df.copy()
        exclude_cols = {date_col, "ticker"}
        value_cols = [c for c in panel_df.columns if c not in exclude_cols]

        grouped = result.groupby(date_col)[value_cols]
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0, np.nan)

        clipped = result[value_cols].clip(lower=means - 3 * stds, upper=means + 3 * stds, axis=0)
        clipped_grouped = clipped.groupby(result[date_col])
        clipped_means = clipped_grouped.transform("mean")
        clipped_stds = clipped_grouped.transform("std").replace(0, np.nan)

        standardized = ((clipped - clipped_means) / clipped_stds).fillna(0.0)
        result[value_cols] = standardized
        return result

    def __repr__(self) -> str:  # noqa: D105
        return "CharacteristicBuilder()"
