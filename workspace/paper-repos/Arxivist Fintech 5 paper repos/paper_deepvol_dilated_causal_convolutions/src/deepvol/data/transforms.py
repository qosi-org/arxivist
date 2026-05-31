"""
Data transforms: log-return computation and sequence preparation.
Implements Eq. 1 (Section 2.1): r_{i,t} = log(p_{i,t} / p_{i-1,t})
"""
import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Eq. 1: log returns from price series."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_realised_variance(intraday_returns: np.ndarray) -> float:
    """Eq. 2: RV_t = sum_{i=1}^{I} r_{i,t}^2"""
    return float(np.sum(intraday_returns ** 2))


def build_intraday_sequences(
    returns_df: pd.DataFrame,
    conditioning_range: int,
    intervals_per_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) pairs for DeepVol.
    X: [N, 1, T*J]  — raw intraday returns over T past days
    y: [N, 1]        — realised variance for the forecast day
    """
    trading_days = returns_df.index.get_level_values("date").unique().sort_values()
    X_list, y_list = [], []

    for i in range(conditioning_range, len(trading_days) - 1):
        past_days = trading_days[i - conditioning_range: i]
        forecast_day = trading_days[i]

        past_returns = returns_df.loc[past_returns_idx(returns_df, past_days)]
        seq = past_returns.values.flatten().astype(np.float32)
        if len(seq) != conditioning_range * intervals_per_day:
            continue  # skip incomplete days

        rv_day = returns_df.loc[rv_idx(returns_df, forecast_day)].values.flatten()
        rv = compute_realised_variance(rv_day)

        X_list.append(seq[np.newaxis, :])   # [1, T*J]
        y_list.append([rv])

    X = np.stack(X_list)[:, np.newaxis, :]  # [N, 1, T*J]
    y = np.array(y_list, dtype=np.float32)   # [N, 1]
    return X, y


def past_returns_idx(df, days):
    return df.index.get_level_values("date").isin(days)

def rv_idx(df, day):
    return df.index.get_level_values("date") == day
