"""
forecast_risk.data.spf_loader
==============================
Survey of Professional Forecasters (SPF) data loader.

Paper: Section 3 — Models (SPF competitor)
"Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

Paper description:
  "The median aggregates predictions from dozens of professionals with access
  to proprietary models, real-time data, and qualitative judgment unavailable
  to statistical methods." (Stark 2010; Engelberg et al. 2009)

SPF data source:
  Federal Reserve Bank of Philadelphia
  https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/
  survey-of-professional-forecasters

Targets used in the paper (Sec 3):
  - GDP growth (RGDP)
  - CPI inflation (CPI)
  - Unemployment rate (UNEMP)
  - Housing starts (HOUSING)

Horizons evaluated: h = 1, 2, 4 quarters ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# Map paper target names to SPF series identifiers
SPF_SERIES_MAP = {
    "gdp_growth":       "RGDP",
    "cpi_inflation":    "CPI",
    "unemployment_rate": "UNEMP",
    "housing_starts":   "HOUSING",
}


class SPFLoader:
    """
    Loads and aligns SPF median forecasts with FRED-QD evaluation periods.

    The SPF provides median point forecasts at horizons h=1,2,4 quarters ahead.
    Forecasts are matched to FRED-QD dates for direct comparison with ML models.

    Args:
        spf_path:  Path to the SPF median CSV file (downloaded from Philadelphia Fed).
    """

    def __init__(self, spf_path: str | Path):
        self.spf_path = Path(spf_path)
        self._data: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        """Load and parse SPF CSV file."""
        if self._data is not None:
            return self._data

        if not self.spf_path.exists():
            raise FileNotFoundError(
                f"SPF file not found: {self.spf_path}\n"
                "Download from: https://www.philadelphiafed.org/surveys-and-data/"
                "real-time-data-research/survey-of-professional-forecasters"
            )

        # Philadelphia Fed SPF format: YEAR, QUARTER, series columns
        df = pd.read_csv(self.spf_path)

        # Normalize column names
        df.columns = [c.strip().upper() for c in df.columns]

        # Build quarterly PeriodIndex
        if "YEAR" in df.columns and "QUARTER" in df.columns:
            df["date"] = pd.PeriodIndex(
                [f"{int(y)}Q{int(q)}" for y, q in zip(df["YEAR"], df["QUARTER"])],
                freq="Q"
            )
            df = df.set_index("date")
        elif "DATE" in df.columns:
            df["date"] = pd.PeriodIndex(df["DATE"], freq="Q")
            df = df.set_index("date")
        else:
            raise ValueError(
                "Cannot parse SPF date columns. Expected 'YEAR'+'QUARTER' or 'DATE'."
            )

        self._data = df
        return self._data

    def get_median_forecasts(
        self,
        target: str,
        horizon: int,
    ) -> pd.Series:
        """
        Get SPF median point forecast series for a given target and horizon.

        The paper uses SPF median forecasts at h=1,2,4 quarters ahead.
        The forecast for period t+h made at period t is matched to the
        realization at t+h.

        Args:
            target:  Target name ('gdp_growth', 'cpi_inflation', etc.).
            horizon: Forecast horizon (1, 2, or 4).

        Returns:
            Series indexed by forecast-made date (PeriodIndex), values = median forecast.

        Note:
            The exact column naming in Philadelphia Fed files varies by vintage.
            Common patterns:
              - RGDP1, RGDP2, RGDP4  (h=1,2,4 for real GDP)
              - CPI1, CPI2, CPI4
              - UNEMP1, UNEMP2, UNEMP4
        """
        df = self._load()
        spf_series = SPF_SERIES_MAP.get(target)

        if spf_series is None:
            raise ValueError(
                f"Unknown target '{target}'. "
                f"Available: {list(SPF_SERIES_MAP.keys())}"
            )

        # Try common column name patterns
        col_candidates = [
            f"{spf_series}{horizon}",       # e.g. RGDP1
            f"{spf_series}_{horizon}",      # e.g. RGDP_1
            f"{spf_series}H{horizon}",      # e.g. RGDPH1
            f"{spf_series}Q{horizon}",      # e.g. RGDPQ1
        ]

        col = None
        for candidate in col_candidates:
            if candidate in df.columns:
                col = candidate
                break

        if col is None:
            available_cols = [c for c in df.columns if spf_series in c]
            raise KeyError(
                f"Cannot find SPF column for target='{target}', horizon={horizon}.\n"
                f"Tried: {col_candidates}\n"
                f"Columns containing '{spf_series}': {available_cols}\n"
                f"All columns: {list(df.columns[:20])}..."
            )

        return df[col].dropna()

    def get_spf_losses(
        self,
        target: str,
        horizon: int,
        realized: pd.Series,
        loss_fn: str = "squared_error",
    ) -> np.ndarray:
        """
        Compute SPF forecast losses aligned to realized values.

        Args:
            target:    Target variable name.
            horizon:   Forecast horizon.
            realized:  Realized values series (PeriodIndex, same as FRED-QD dates).
            loss_fn:   Loss function ('squared_error' or 'absolute_error').

        Returns:
            Loss array aligned to realized.index. NaN where SPF forecast unavailable.
        """
        forecasts = self.get_median_forecasts(target, horizon)

        losses = np.full(len(realized), np.nan)
        for i, (date, y_true) in enumerate(realized.items()):
            # SPF forecast made at date-horizon for date
            forecast_date = date - horizon
            if forecast_date in forecasts.index:
                y_hat = float(forecasts[forecast_date])
                if loss_fn == "squared_error":
                    losses[i] = (y_true - y_hat) ** 2
                elif loss_fn == "absolute_error":
                    losses[i] = abs(y_true - y_hat)

        return losses
