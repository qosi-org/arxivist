"""
Synthetic panel generator, used to smoke-test the full pipeline end-to-end
without the real (proprietary) China A-share dataset described in Sec 3.1-3.2.

This does NOT attempt to mimic the real data's statistical properties (factor
premia, cross-sectional correlation structure, survivorship dynamics) -- it
exists purely so every module in this repo (universe construction, feature
selection, quantum/classical kernels, KRR, walk-forward evaluation) can be
exercised and unit-tested without access to a real vendor feed. See
data/README_data.md for what a real data integration needs to provide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qkernel_finance.data.characteristics import MARKET_CHARACTERISTICS, FUNDAMENTAL_CHARACTERISTICS


def make_synthetic_panel(
    num_days: int = 300,
    num_tickers: int = 60,
    seed: int = 42,
    extra_diagnostic_chars: bool = False,
) -> pd.DataFrame:
    """Generates a synthetic daily standardized-characteristic + forward-return panel.

    Args:
        num_days: number of trading days to simulate.
        num_tickers: number of synthetic tickers.
        seed: RNG seed.
        extra_diagnostic_chars: if True, also generates the 4 diagnostic-study-only
            characteristics (RSRS, analyst_revisions, eps_vol, and one variant)
            needed for the 31-characteristic short-sample study (Sec 3.1) and
            the interaction catalog's momentum_x_revisions / momentum_x_uncertainty
            / surprise_x_uncertainty entries.

    Returns:
        DataFrame with columns [date, ticker, <characteristics...>, price,
        float_mktcap, fwd_return_20d], already cross-sectionally standardized
        for the characteristic columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=num_days)
    tickers = [f"SYN{i:04d}.SZ" for i in range(num_tickers)]

    characteristics = list(MARKET_CHARACTERISTICS) + list(FUNDAMENTAL_CHARACTERISTICS)
    if extra_diagnostic_chars:
        characteristics += ["rsrs", "analyst_revisions", "eps_vol", "rsrs_variant"]

    rows = []
    # Give each ticker a fixed latent "quality" so returns have *some* structure
    # to predict (otherwise every model's IC would trivially be ~0 with huge variance).
    latent_quality = rng.normal(0, 1, size=num_tickers)

    for date in dates:
        char_values = rng.normal(0, 1, size=(num_tickers, len(characteristics)))
        # Inject a small amount of true signal from the first characteristic into returns.
        signal = 0.02 * char_values[:, 0] + 0.01 * latent_quality
        noise = rng.normal(0, 0.05, size=num_tickers)
        fwd_return = signal + noise

        price = np.clip(10 + rng.normal(0, 1, size=num_tickers).cumsum() * 0.01, 1, None)
        float_mktcap = np.abs(rng.normal(1e9, 3e8, size=num_tickers))

        df = pd.DataFrame(char_values, columns=characteristics)
        df["date"] = date
        df["ticker"] = tickers
        df["price"] = price
        df["float_mktcap"] = float_mktcap
        df["fwd_return_20d"] = fwd_return
        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)

    # Cross-sectional standardization (winsorize +/-3 sigma, standardize per
    # day) -- reuses CharacteristicBuilder.standardize() rather than
    # duplicating the logic, so both paths share one (tested) implementation.
    from qkernel_finance.data.characteristics import CharacteristicBuilder

    non_char_cols = {"date", "ticker", "price", "float_mktcap", "fwd_return_20d"}
    char_only_cols = [c for c in characteristics if c not in non_char_cols]
    builder = CharacteristicBuilder()
    standardized_chars = builder.standardize(panel[["date", "ticker"] + char_only_cols], date_col="date")
    panel[char_only_cols] = standardized_chars[char_only_cols]

    return panel.reset_index(drop=True)
