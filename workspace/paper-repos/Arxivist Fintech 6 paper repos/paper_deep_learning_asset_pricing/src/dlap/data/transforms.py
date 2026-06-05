"""
data/transforms.py — Data preprocessing transforms for CRSP panel data.

Implements cross-sectional quantile normalization (Section V.A):
"For each characteristic variable in each month, we rank them cross-sectionally
and convert them into quantiles."

Also implements macroeconomic variable transformations per McCracken & Ng (2016).

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Section V.A.
"""

import numpy as np
from typing import Optional


def cross_sectional_quantile_normalize(
    chars: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    """
    Cross-sectionally rank-normalize firm characteristics to quantiles.

    For each month t and characteristic k:
        rank stocks by chars[t, :, k]
        normalize to [0, 1] via rank / (N + 1)
        optionally center by subtracting 0.5

    Used for all 46 firm characteristics (Section V.A).

    Args:
        chars: [T, N, K] raw firm characteristics
        center: if True, subtract cross-sectional mean (paper: center around mean)

    Returns:
        normalized: [T, N, K] quantile-normalized characteristics in [-0.5, 0.5]
    """
    T, N, K = chars.shape
    normalized = np.zeros_like(chars, dtype=np.float32)

    for t in range(T):
        for k in range(K):
            col = chars[t, :, k]
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() < 2:
                continue
            ranks = np.zeros(N, dtype=np.float32)
            valid_vals = col[valid_mask]
            # Rank valid values; ties get average rank
            from scipy.stats import rankdata
            ranks[valid_mask] = rankdata(valid_vals, method="average") / (valid_mask.sum() + 1)
            normalized[t, :, k] = ranks

    if center:
        normalized -= normalized.mean(axis=1, keepdims=True)

    return normalized


def apply_macro_transforms(
    macro: np.ndarray,
    t_codes: np.ndarray,
) -> np.ndarray:
    """
    Apply standard transformations to macroeconomic time series.

    Transformation codes from McCracken & Ng (2016) and Table A.VI:
        1: no transformation (xt)
        2: first difference (Δxt)
        3: second difference (Δ²xt)
        4: log (log(xt))
        5: first difference of log (Δlog(xt))
        6: second difference of log (Δ²log(xt))
        7: Δ(xt/xt-1 - 1.0)

    Args:
        macro: [T, M] raw macroeconomic time series
        t_codes: [M] integer transformation codes

    Returns:
        transformed: [T, M] transformed time series (same shape, NaN at boundaries)
    """
    T, M = macro.shape
    transformed = np.full_like(macro, np.nan, dtype=np.float32)

    for m in range(M):
        x = macro[:, m].astype(np.float64)
        code = int(t_codes[m])

        if code == 1:
            transformed[:, m] = x
        elif code == 2:
            transformed[1:, m] = np.diff(x)
        elif code == 3:
            transformed[2:, m] = np.diff(x, n=2)
        elif code == 4:
            with np.errstate(invalid="ignore"):
                transformed[:, m] = np.where(x > 0, np.log(x), np.nan)
        elif code == 5:
            with np.errstate(invalid="ignore"):
                lx = np.where(x > 0, np.log(x), np.nan)
            transformed[1:, m] = np.diff(lx)
        elif code == 6:
            with np.errstate(invalid="ignore"):
                lx = np.where(x > 0, np.log(x), np.nan)
            transformed[2:, m] = np.diff(lx, n=2)
        elif code == 7:
            ratio = x[1:] / x[:-1] - 1.0
            transformed[2:, m] = np.diff(ratio)
        else:
            transformed[:, m] = x

    # Forward-fill NaN at boundaries
    for m in range(M):
        nan_mask = np.isnan(transformed[:, m])
        if nan_mask.any():
            transformed[nan_mask, m] = 0.0  # zero-pad boundaries

    return transformed
