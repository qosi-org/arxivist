"""
pricing/black_scholes.py
────────────────────────
Black-Scholes formula, implied volatility inversion (Brent's method), and Vega.
Used for:
  - Inverting MC option prices to implied volatilities
  - Computing inverse-Vega weights γ_i for the calibration loss (Section 4.2)
  - Analytical IV surface computation (Sections 2.1, 2.2)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


class BlackScholes:
    """
    Static collection of Black-Scholes pricing and calibration utilities.

    Paper references:
      - Section 2 (passim): BS formula as the inversion target for implied vol
      - Section 4.2: γ_i = inverse Vega weighting
    """

    @staticmethod
    def d_plus(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """d+ = [log(S/K) + (r + σ²/2)T] / (σ√T)"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d_minus(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """d- = d+ - σ√T"""
        dp = BlackScholes.d_plus(S, K, T, r, sigma)
        return dp - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes European call price.
        C = S·Φ(d+) - K·e^{-rT}·Φ(d-)

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (years).
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            Call option price.
        """
        if T <= 0.0:
            return max(S - K * np.exp(-r * T), 0.0)
        dp = BlackScholes.d_plus(S, K, T, r, sigma)
        dm = dp - sigma * np.sqrt(T)
        return S * norm.cdf(dp) - K * np.exp(-r * T) * norm.cdf(dm)

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes Vega = ∂C/∂σ = S·√T·φ(d+)
        Used to compute inverse-Vega weights γ_i = 1/Vega_i.  (Section 4.2)

        Returns:
            Vega of the call option.
        """
        if T <= 0.0 or sigma <= 0.0:
            return 1e-12  # avoid division by zero in weights
        dp = BlackScholes.d_plus(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(dp)

    @staticmethod
    def implied_vol(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma_lo: float = 1e-6,
        sigma_hi: float = 10.0,
        tol: float = 1e-10,
    ) -> float:
        """
        Invert Black-Scholes to find implied volatility via Brent's method.

        Args:
            price: Observed option price (market or model).
            S:     Current stock price.
            K:     Strike.
            T:     Maturity (years).
            r:     Risk-free rate.
            sigma_lo: Lower bound for root search.
            sigma_hi: Upper bound for root search.
            tol: Numerical tolerance.

        Returns:
            Implied volatility σ such that BS(S,K,T,r,σ) = price.
            Returns NaN if Brent fails (e.g. price outside no-arbitrage bounds).
        """
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
        if price <= intrinsic + 1e-12:
            return float("nan")

        def objective(sigma: float) -> float:
            return BlackScholes.call_price(S, K, T, r, sigma) - price

        try:
            return brentq(objective, sigma_lo, sigma_hi, xtol=tol)
        except ValueError:
            return float("nan")

    @staticmethod
    def implied_vol_surface(
        prices: np.ndarray,  # shape [nT, nK]
        S: float,
        strikes: np.ndarray,  # shape [nK]
        maturities: np.ndarray,  # shape [nT]
        r: float,
    ) -> np.ndarray:
        """
        Invert an entire option price surface to an implied volatility surface.

        Args:
            prices:     [nT, nK] array of call prices.
            S:          Current stock price.
            strikes:    [nK] array of strikes.
            maturities: [nT] array of maturities.
            r:          Risk-free rate.

        Returns:
            [nT, nK] array of implied volatilities.
        """
        nT, nK = prices.shape
        assert len(maturities) == nT and len(strikes) == nK, (
            f"Shape mismatch: prices={prices.shape}, "
            f"maturities={len(maturities)}, strikes={len(strikes)}"
        )
        iv = np.full((nT, nK), float("nan"))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                iv[i, j] = BlackScholes.implied_vol(prices[i, j], S, K, T, r)
        return iv

    @staticmethod
    def inverse_vega_weights(
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        sigma0: float,
    ) -> np.ndarray:
        """
        Compute inverse-Vega weights γ_i = 1/Vega_i for all (K,T) pairs.
        Vega is evaluated at Black-Scholes with σ = σ₀ (initial volatility).

        Used in the calibration loss L(ℓ) = Σ γ_i (C_mkt - C_model)².  (Section 4.2)

        Args:
            S:          Stock price.
            strikes:    [nK] array of strikes.
            maturities: [nT] array of maturities.
            r:          Risk-free rate.
            sigma0:     Volatility at which to evaluate Vega (use market σ₀).

        Returns:
            [nT*nK] flattened array of inverse-Vega weights.
        """
        weights = []
        for T in maturities:
            for K in strikes:
                v = BlackScholes.vega(S, K, T, r, sigma0)
                weights.append(1.0 / max(v, 1e-12))
        return np.array(weights)
