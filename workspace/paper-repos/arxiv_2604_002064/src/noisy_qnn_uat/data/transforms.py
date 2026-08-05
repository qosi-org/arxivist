"""Input normalisation and the classical closed-form Black-Scholes Put formula.

Implements: Architecture Plan module `data/transforms.py`.
Paper sections: Eq. (4.1) (input normalisation to [0,1]^d), Section 2.3.3
(Black-Scholes Put closed-form price, used as ground truth / classical baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class InputNormalizer:
    """Normalises inputs to [0,1]^d per Eq. (4.1), clamping out-of-range values."""

    def normalize(self, x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """Normalise x to [0,1]^d via (x - x_min) / (x_max - x_min), then clamp.

        Any value outside the training range [x_min, x_max] is clamped to [0, 1]
        to prevent extrapolation artefacts (Eq. 4.1, "Any value outside the
        training range is clamped to [0, 1]").

        Args:
            x: input array, shape [..., d].
            x_min: per-dimension minimum, shape [d].
            x_max: per-dimension maximum, shape [d].

        Returns:
            Normalised array in [0, 1]^d, same shape as x.
        """
        assert x.shape[-1] == x_min.shape[-1] == x_max.shape[-1], (
            f"Dimension mismatch: x={x.shape}, x_min={x_min.shape}, x_max={x_max.shape}"
        )
        denom = x_max - x_min
        assert np.all(denom != 0), "x_max - x_min contains zero(s); cannot normalise"
        x_norm = (x - x_min) / denom
        return np.clip(x_norm, 0.0, 1.0)

    def denormalize(self, x_norm: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """Inverse of `normalize`, for converting normalised inputs back to raw units."""
        return x_norm * (x_max - x_min) + x_min


def black_scholes_put_price(
    S0: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Classical Black-Scholes European Put price (Section 2.3.3).

    PutBS(S0,K,T,r,sigma) = K*N(-d-) - S0*N(-d+)
    d+/- = [log(S0/K) +/- (1/2)*sigma^2*T] / (sigma*sqrt(T)) ... adjusted by r

    NOTE: the paper's worked examples in Section 2.3 assume no interest rate (r=0)
    for the pure Fourier-transform derivations, but Section 4.3's numerical
    experiments sweep r in [0.02, 0.05] as an input dimension. This implementation
    includes the standard risk-neutral discounting term so that r behaves as an
    actual pricing input consistent with the 5-dimensional experiment of Section 4.3.

    Args:
        S0: spot price(s).
        K: strike price(s).
        T: time to maturity (years).
        r: risk-free interest rate.
        sigma: volatility.

    Returns:
        The Black-Scholes European Put price (scalar or array, matching inputs).
    """
    S0 = np.asarray(S0, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    sqrt_t = np.sqrt(T)
    d_plus = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    d_minus = d_plus - sigma * sqrt_t

    put = K * np.exp(-r * T) * norm.cdf(-d_minus) - S0 * norm.cdf(-d_plus)
    return put


def gaussian_density(x: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian density f_sigma(x) = 1/(sigma*sqrt(2*pi)) * exp(-x^2 / (2*sigma^2)).

    Paper section: Section 2.3.1 "Gaussian density".

    Args:
        x: input array, any shape.
        sigma: standard deviation.

    Returns:
        f_sigma(x), same shape as x.
    """
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-(x ** 2) / (2.0 * sigma ** 2))
