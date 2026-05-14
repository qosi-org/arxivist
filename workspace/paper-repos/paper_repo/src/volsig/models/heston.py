"""
models/heston.py
────────────────
Heston stochastic volatility model:
  (1) Market price simulation (for generating synthetic IV surfaces)
  (2) Analytical second-order ASV calibration  (Section 2.1, Alòs et al. 2015)

Paper equations implemented:
  - Eq. (2.1)–(2.2):  Heston stock + variance SDEs
  - Eq. (2.4):  Short-T implied vol expansion
  - Eq. (2.5):  Long-T implied vol expansion
  - Eq. (2.6):  ATM implied vol expansion
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
from typing import Dict, Optional, Tuple

from volsig.pricing.black_scholes import BlackScholes


class HestonModel:
    """
    Heston stochastic volatility model for market price generation and
    analytical implied volatility calibration via the ASV expansion.

    Model dynamics (Eqs. 2.1–2.2):
        dS_t = r S_t dt + σ_t S_t d(ρW_t + √(1-ρ²)B_t)
        dσ²_t = κ(θ - σ²_t) dt + ν √(σ²_t) dW_t

    Calibration via second-order expansion from Alòs et al. (2015),
    using three asymptotic regimes (Section 2.1):
        - ATM term structure (Eq. 2.6)
        - Short-maturity skew (Eq. 2.4)
        - Long-maturity level (Eq. 2.5)
    """

    def __init__(
        self,
        sigma0: float,
        nu: float,
        kappa: float,
        theta: float,
        rho: float,
        S0: float = 100.0,
        r: float = 0.0,
    ):
        """
        Args:
            sigma0: Initial volatility σ₀.
            nu:     Vol-of-vol ν.
            kappa:  Mean-reversion speed κ.
            theta:  Long-run variance θ.
            rho:    Correlation ρ ∈ (-1,1).
            S0:     Initial stock price.
            r:      Risk-free rate.
        """
        self.sigma0 = sigma0
        self.nu = nu
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.S0 = S0
        self.r = r

    def __repr__(self) -> str:
        return (f"HestonModel(σ₀={self.sigma0}, ν={self.nu}, κ={self.kappa}, "
                f"θ={self.theta}, ρ={self.rho})")

    # ── Market simulation ────────────────────────────────────────────────────

    def simulate(
        self,
        nMC: int,
        T_steps: int,
        dt: float,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Euler-Maruyama simulation of the Heston model.

        Args:
            nMC:     Number of Monte Carlo paths.
            T_steps: Number of time steps.
            dt:      Time step size.
            seed:    Random seed.

        Returns:
            S: [nMC, T_steps+1] stock price paths.
            V: [nMC, T_steps+1] variance paths.
        """
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((nMC, T_steps)) * np.sqrt(dt)  # dW ~ N(0,dt)
        Z = rng.standard_normal((nMC, T_steps)) * np.sqrt(dt)  # independent

        S = np.zeros((nMC, T_steps + 1), dtype=np.float64)
        V = np.zeros((nMC, T_steps + 1), dtype=np.float64)
        S[:, 0] = self.S0
        V[:, 0] = self.sigma0 ** 2

        for t in range(T_steps):
            V_pos = np.abs(V[:, t])  # reflection scheme
            # Correlated BM for stock: ρW + √(1-ρ²)Z
            dS_noise = self.rho * W[:, t] + np.sqrt(1.0 - self.rho ** 2) * Z[:, t]
            sigma_t = np.sqrt(V_pos)
            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - 0.5 * V_pos) * dt + sigma_t * dS_noise
            )
            # Variance SDE (Eq. 2.2)
            V[:, t + 1] = (V_pos
                           + self.kappa * (self.theta - V_pos) * dt
                           + self.nu * sigma_t * W[:, t])
            V[:, t + 1] = np.abs(V[:, t + 1])

        return S, V

    def price_call_MC(
        self,
        K: float,
        T: float,
        nMC: int = 200_000,
        T_steps: Optional[int] = None,
        dt: float = 1.0 / 252,
        seed: Optional[int] = None,
    ) -> float:
        """Price a European call by Monte Carlo simulation."""
        if T_steps is None:
            T_steps = max(int(T / dt), 1)
        dt_actual = T / T_steps
        S, _ = self.simulate(nMC, T_steps, dt_actual, seed=seed)
        payoffs = np.maximum(S[:, -1] - K, 0.0) * np.exp(-self.r * T)
        return float(np.mean(payoffs))

    def implied_vol_surface_MC(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        nMC: int = 200_000,
        dt: float = 1.0 / 252,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the implied volatility surface by Monte Carlo.

        Returns:
            IV: [nT, nK] implied volatility surface.
        """
        nT = len(maturities)
        nK = len(strikes)
        prices = np.zeros((nT, nK))
        for i, T in enumerate(maturities):
            T_steps = max(int(T / dt), 1)
            dt_actual = T / T_steps
            S, _ = self.simulate(nMC, T_steps, dt_actual, seed=seed)
            for j, K in enumerate(strikes):
                payoffs = np.maximum(S[:, -1] - K, 0.0) * np.exp(-self.r * T)
                prices[i, j] = float(np.mean(payoffs))
        return BlackScholes.implied_vol_surface(prices, self.S0, strikes, maturities, self.r)

    # ── Analytical ASV expansion  (Section 2.1, Alòs et al. 2015) ───────────

    def iv_atm_approx(self, T: float) -> float:
        """
        ATM implied volatility approximation.  (Eq. 2.6)

        I(T, K*) ≈ σ₀ + [3σ₀²ρν - 6κ(σ₀²-θ) - ν²] / (24σ₀) · T

        Args:
            T: Maturity.

        Returns:
            ATM implied volatility at maturity T.
        """
        σ0, ν, κ, θ, ρ = self.sigma0, self.nu, self.kappa, self.theta, self.rho
        slope = (3 * σ0 ** 2 * ρ * ν - 6 * κ * (σ0 ** 2 - θ) - ν ** 2) / (24 * σ0)
        return σ0 + slope * T

    def iv_short_T_approx(self, x: float, k: float, T: float) -> float:
        """
        Short-maturity implied volatility approximation.  (Eq. 2.4)

        I(0, K) ≈ σ₀ - ρν/(4σ₀) · (x-k) + ν²/(24σ₀³) · (x-k)²

        where x = log S, k = log K.

        Args:
            x: Log stock price.
            k: Log strike.
            T: Maturity (unused in limit, present for interface consistency).

        Returns:
            Implied volatility.
        """
        σ0, ν, ρ = self.sigma0, self.nu, self.rho
        m = x - k  # log-moneyness
        return σ0 - (ρ * ν / (4 * σ0)) * m + (ν ** 2 / (24 * σ0 ** 3)) * m ** 2

    def iv_long_T_approx(self, T: float) -> float:
        """
        Long-maturity ATM implied volatility approximation.  (Eq. 2.5)

        I(T, K*) ≈ √θ · (1 + νρ/(4κ) - ν²/(32κ²))
                   + [σ₀²-θ)/(2κ√θ) + νρ(σ₀²-2θ)/(4κ²√θ) - ν²(σ₀²-5θ/2+4κ)/(32√θκ³)] · 1/T

        Args:
            T: Maturity.

        Returns:
            Long-maturity ATM implied volatility.
        """
        σ0, ν, κ, θ, ρ = self.sigma0, self.nu, self.kappa, self.theta, self.rho
        sqrt_theta = np.sqrt(θ)
        const = sqrt_theta * (1.0 + ν * ρ / (4 * κ) - ν ** 2 / (32 * κ ** 2))
        coeff_1_T = (
            (σ0 ** 2 - θ) / (2 * κ * sqrt_theta)
            + ν * ρ * (σ0 ** 2 - 2 * θ) / (4 * κ ** 2 * sqrt_theta)
            - ν ** 2 * (σ0 ** 2 - 2.5 * θ + 4 * κ) / (32 * sqrt_theta * κ ** 3)
        )
        return const + coeff_1_T / T

    def implied_vol_ASV(
        self,
        K: float,
        T: float,
        regime: str = "auto",
    ) -> float:
        """
        Second-order ASV implied volatility approximation.

        Selects the expansion regime automatically:
          - T < 0.3:  short-T expansion (Eq. 2.4)
          - T > 2.0:  long-T expansion (Eq. 2.5)
          - otherwise: ATM expansion (Eq. 2.6) with skew correction

        Args:
            K:      Strike.
            T:      Maturity.
            regime: 'auto', 'short', 'long', or 'atm'.

        Returns:
            ASV implied volatility approximation.
        """
        x = np.log(self.S0) + self.r * T  # forward log-price
        k = np.log(K)

        if regime == "auto":
            if T < 0.3:
                regime = "short"
            elif T > 2.0:
                regime = "long"
            else:
                regime = "atm"

        if regime == "short":
            return self.iv_short_T_approx(x, k, T)
        elif regime == "long":
            return self.iv_long_T_approx(T)
        else:  # atm + skew
            iv_atm = self.iv_atm_approx(T)
            # Add skew correction proportional to log-moneyness
            m = x - k
            skew = -self.rho * self.nu / (4 * self.sigma0)
            return iv_atm + skew * m

    def implied_vol_surface_ASV(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the full ASV implied volatility surface.

        Returns:
            IV_ASV: [nT, nK] analytical approximation surface.
        """
        nT, nK = len(maturities), len(strikes)
        IV = np.zeros((nT, nK))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                IV[i, j] = self.implied_vol_ASV(K, T)
        return IV

    # ── Parameter calibration from ASV system of equations ───────────────────

    @staticmethod
    def calibrate_from_surface(
        iv_surface: np.ndarray,   # [nT, nK]
        maturities: np.ndarray,   # [nT]
        strikes: np.ndarray,      # [nK]
        S0: float = 100.0,
        r: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calibrate Heston parameters from an implied volatility surface using the
        three-equation system of Section 2.1.

        Steps (Section 2.1):
          1. Fit ATM IV vs T → σ₀ and (3σ₀²ρν - 6κ(σ₀²-θ) - ν²) / (24σ₀)  [Eq. 2.6]
          2. Fit short-T skew vs log-moneyness → ρν  [Eq. 2.4]
          3. Fit long-T ATM IV vs 1/T → √θ(1+νρ/(4κ)-ν²/(32κ²)) and c₁  [Eq. 2.5]
          4. Solve 3-equation system for (σ₀, ν, κ, θ, ρ).

        Args:
            iv_surface: [nT, nK] implied volatility surface.
            maturities: [nT] array of maturities.
            strikes:    [nK] array of strikes.
            S0:         Initial stock price.
            r:          Risk-free rate.

        Returns:
            dict with keys: sigma0, nu, kappa, theta, rho.
        """
        # ATM index
        atm_idx = np.argmin(np.abs(strikes - S0))
        iv_atm = iv_surface[:, atm_idx]  # [nT]

        # Step 1: Linear fit of ATM IV vs T  →  σ₀ (intercept) and slope_atm
        # I(T,K*) ≈ σ₀ + slope_atm·T   (Eq. 2.6)
        T_arr = np.array(maturities)
        A1 = np.column_stack([np.ones_like(T_arr), T_arr])
        coeffs_atm, _, _, _ = np.linalg.lstsq(A1, iv_atm, rcond=None)
        sigma0_est = float(coeffs_atm[0])
        slope_atm = float(coeffs_atm[1])
        # slope_atm = (3σ₀²ρν - 6κ(σ₀²-θ) - ν²) / (24σ₀)

        # Step 2: Skew from short-T smile  →  ρν  (Eq. 2.4)
        short_T_idx = np.argmin(T_arr)
        short_iv = iv_surface[short_T_idx, :]   # [nK]
        x = np.log(S0) + r * T_arr[short_T_idx]
        log_moneyness = x - np.log(strikes)     # [nK]
        A2 = np.column_stack([np.ones_like(log_moneyness), log_moneyness])
        coeffs_skew, _, _, _ = np.linalg.lstsq(A2, short_iv, rcond=None)
        rho_nu_est = float(-coeffs_skew[1]) * 4 * sigma0_est
        # I ≈ σ₀ + (-ρν/(4σ₀))·m  →  skew coeff = -ρν/(4σ₀)  →  ρν = -4σ₀·skew

        # Step 3: Long-T level  →  √θ·A_long and coeff_1/T  (Eq. 2.5)
        long_T_mask = T_arr > 1.0
        if long_T_mask.sum() >= 2:
            inv_T = 1.0 / T_arr[long_T_mask]
            iv_long = iv_atm[long_T_mask]
            A3 = np.column_stack([np.ones_like(inv_T), inv_T])
            coeffs_long, _, _, _ = np.linalg.lstsq(A3, iv_long, rcond=None)
            sqrt_theta_A = float(coeffs_long[0])  # √θ·(1 + νρ/(4κ) - ν²/(32κ²))
        else:
            sqrt_theta_A = np.sqrt(sigma0_est ** 2)  # fallback

        # Solve for parameters from the three extracted quantities.
        # This is a simplified solver that assumes small ν and ρ corrections
        # are second-order, giving a closed-form estimate.
        theta_est = sqrt_theta_A ** 2  # 0th-order: ignore correction terms
        rho_est = rho_nu_est / max(abs(rho_nu_est) ** 0.5, 1e-6)  # rough estimate
        nu_est = abs(rho_nu_est / max(abs(rho_est), 1e-6))
        # Recover κ from the ATM slope equation:
        # slope_atm = (3σ₀²ρν - 6κ(σ₀²-θ) - ν²) / (24σ₀)
        # → κ = (3σ₀²ρν - ν² - 24σ₀·slope_atm) / (6(σ₀²-θ))
        denom = 6 * (sigma0_est ** 2 - theta_est)
        if abs(denom) > 1e-8:
            kappa_est = (3 * sigma0_est ** 2 * rho_est * nu_est
                         - nu_est ** 2
                         - 24 * sigma0_est * slope_atm) / denom
        else:
            kappa_est = 3.0  # fallback

        return {
            "sigma0": sigma0_est,
            "nu": max(nu_est, 1e-4),
            "kappa": max(kappa_est, 1e-4),
            "theta": max(theta_est, 1e-4),
            "rho": float(np.clip(rho_est, -0.999, 0.999)),
        }
