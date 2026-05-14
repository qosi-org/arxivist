"""
models/rough_bergomi.py
───────────────────────
Rough Bergomi model simulation and the new VIX-based analytical calibration.

Paper sections:
  - Section 2.2:  VIX-based calibration procedure (Steps 1–4)
  - Eq. (2.7):    Rough Bergomi model definition
  - Section 6:    Numerical experiments

Steps implemented:
  Step 1 — H estimation from implied vol skew at two maturities
  Step 2 — η estimation from VIX ATMI short-time asymptotics
  Step 3 — ρ estimation from short-time skew of ATMI
  Step 4 — σ₀ regression from ATM term structure
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from volsig.pricing.black_scholes import BlackScholes


# ─────────────────────────────────────────────────────────────────────────────
# Rough Bergomi model simulation
# ─────────────────────────────────────────────────────────────────────────────

class RoughBergomiModel:
    """
    Rough Bergomi model (Eq. 2.7):

        dS_t = r S_t dt + σ_t S_t d(ρW_t + √(1-ρ²)B_t)
        σ²_t = σ₀² exp(η W^H_t - ½ η² t^{2H})

    where W^H_t = ∫₀ᵗ K_H(t,s) dW_s with K_H(t,s) = √(2H) (t-s)^{H-1/2}.

    Used for generating the synthetic market option prices in Section 6.
    """

    def __init__(
        self,
        sigma0: float,
        H: float,
        eta: float,
        rho: float,
        S0: float = 100.0,
        r: float = 0.0,
    ):
        """
        Args:
            sigma0: Initial volatility σ₀.
            H:      Hurst parameter H ∈ (0, 0.5).
            eta:    Vol-of-vol η > 0.
            rho:    Correlation ρ.
            S0:     Initial stock price.
            r:      Risk-free rate.
        """
        assert 0.0 < H < 1.0, f"H must be in (0,1), got {H}"
        assert eta > 0, f"eta must be positive, got {eta}"
        self.sigma0 = sigma0
        self.H = H
        self.eta = eta
        self.rho = rho
        self.S0 = S0
        self.r = r

    def __repr__(self) -> str:
        return (f"RoughBergomiModel(σ₀={self.sigma0}, H={self.H}, "
                f"η={self.eta}, ρ={self.rho})")

    def _simulate_fractional_BM(
        self,
        nMC: int,
        T_steps: int,
        dt: float,
        W_increments: np.ndarray,   # [nMC, T_steps] BM increments
    ) -> np.ndarray:
        """
        Simulate W^H_t via Volterra Riemann sum.  (Eq. 2.7 / Eq. 6.1)

        W^H_{t_n} ≈ √(2H) Σ_{j=0}^{n-1} (t_n - t_j)^{H-1/2} ΔW_j

        Returns:
            fBM: [nMC, T_steps+1] with fBM[:,0] = 0.
        """
        sqrt2H = np.sqrt(2.0 * self.H)
        times = np.arange(T_steps) * dt  # t_j for j=0,...,T-1
        fBM = np.zeros((nMC, T_steps + 1), dtype=np.float64)
        for n in range(1, T_steps + 1):
            t_n = n * dt
            lag = t_n - times[:n]
            kernel = sqrt2H * lag ** (self.H - 0.5)
            fBM[:, n] = W_increments[:, :n] @ kernel
        return fBM

    def simulate(
        self,
        nMC: int,
        T_steps: int,
        dt: float,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Euler simulation of the rough Bergomi model.

        Returns:
            S: [nMC, T_steps+1] stock price paths.
            V: [nMC, T_steps+1] variance paths.
        """
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((nMC, T_steps)) * np.sqrt(dt)
        Z = rng.standard_normal((nMC, T_steps)) * np.sqrt(dt)

        fBM = self._simulate_fractional_BM(nMC, T_steps, dt, W)
        time_grid = np.arange(T_steps + 1) * dt  # [T+1]

        # Variance process: σ²_t = σ₀² exp(η W^H_t - ½ η² t^{2H})  (Eq. 2.7)
        V = self.sigma0 ** 2 * np.exp(
            self.eta * fBM - 0.5 * self.eta ** 2 * time_grid[None, :] ** (2 * self.H)
        )

        S = np.zeros((nMC, T_steps + 1), dtype=np.float64)
        S[:, 0] = self.S0
        for t in range(T_steps):
            sigma_t = np.sqrt(np.maximum(V[:, t], 0.0))
            dS_noise = self.rho * W[:, t] + np.sqrt(1.0 - self.rho ** 2) * Z[:, t]
            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - 0.5 * V[:, t]) * dt + sigma_t * dS_noise
            )

        return S, V

    def implied_vol_surface_MC(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        nMC: int = 100_000,
        dt: float = 1.0 / 252,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the implied volatility surface by Monte Carlo."""
        nT, nK = len(maturities), len(strikes)
        prices = np.zeros((nT, nK))
        for i, T in enumerate(maturities):
            T_steps = max(int(T / dt), 1)
            dt_actual = T / T_steps
            S, _ = self.simulate(nMC, T_steps, dt_actual, seed=seed)
            for j, K in enumerate(strikes):
                payoffs = np.maximum(S[:, -1] - K, 0.0) * np.exp(-self.r * T)
                prices[i, j] = float(np.mean(payoffs))
        return BlackScholes.implied_vol_surface(prices, self.S0, strikes, maturities, self.r)


# ─────────────────────────────────────────────────────────────────────────────
# VIX-based calibration  (Section 2.2)
# ─────────────────────────────────────────────────────────────────────────────

class RoughBergomiVIXCalibrator:
    """
    Closed-form calibration of the rough Bergomi model via short-time asymptotics
    and VIX implied volatility information.  (Section 2.2)

    The four-step procedure estimates (H, η, ρ, σ₀) sequentially:

    Step 1 — H from skew ratio at two maturities  (Alòs et al. 2025 result)
    Step 2 — η from VIX ATMI as T→0  (Alòs & Garcia Lorite 2025, Example 10.2.3)
    Step 3 — ρ from short-time ATMI skew  (Alòs et al. 2024, Section 5.2)
    Step 4 — σ₀ by regression of ATM IV vs T^{2H}
    """

    def __init__(
        self,
        Delta_trading_days: int = 30,  # paper: Section 2.2 explicit
        S0: float = 100.0,
        r: float = 0.0,
    ):
        """
        Args:
            Delta_trading_days: VIX window Δ in trading days (paper: 30).
            S0: Initial stock price.
            r:  Risk-free rate.
        """
        self.Delta = Delta_trading_days / 252.0  # convert to years
        self.S0 = S0
        self.r = r

    def __repr__(self) -> str:
        return f"RoughBergomiVIXCalibrator(Δ={self.Delta:.4f}yr)"

    def estimate_H(
        self,
        iv_surface: np.ndarray,   # [nT, nK]
        maturities: np.ndarray,   # [nT]
        strikes: np.ndarray,      # [nK]
        T1_idx: int = 0,
        T2_idx: int = -1,
    ) -> float:
        """
        Step 1: Estimate Hurst index H from skew at two maturities.  (Section 2.2)

        Uses the formula (Alòs et al. 2025):

            Ĥ = -½ + log( [I(T₁,K⁺)-I(T₁,K⁻)] / [I(T₂,K⁺)-I(T₂,K⁻)] · I²(T₂,K*)/I²(T₁,K*) )
                       / log(T₁/T₂)

        where K⁺_T satisfies d₊=0 (i.e. K⁺_T = S₀·exp((r + σ²/2)T) ≈ S₀·e^{rT})
        and K⁻_T satisfies d₋=0 (i.e. K⁻_T = S₀·exp((r - σ²/2)T)).

        Args:
            iv_surface: [nT, nK] IV surface.
            maturities: [nT] maturities.
            strikes:    [nK] strikes.
            T1_idx:     Index of first (shorter) maturity.
            T2_idx:     Index of second (longer) maturity.

        Returns:
            Ĥ: Estimated Hurst parameter.
        """
        T1 = maturities[T1_idx]
        T2 = maturities[T2_idx]
        assert T1 != T2, "T1 and T2 must be different maturities"

        # ATM IV at each maturity (K* = S0 approximately)
        atm_idx = np.argmin(np.abs(strikes - self.S0))
        iv_atm_T1 = iv_surface[T1_idx, atm_idx]
        iv_atm_T2 = iv_surface[T2_idx, atm_idx]

        # Skew = I(T,K⁺) - I(T,K⁻): use the outermost strikes as proxies
        # (In practice K⁺,K⁻ are defined by d±=0; here we use first/last strikes)
        # TODO: verify exact K±_T definition from paper for a precise implementation
        # For now use finite difference of IV at outermost available strikes
        skew_T1 = iv_surface[T1_idx, -1] - iv_surface[T1_idx, 0]
        skew_T2 = iv_surface[T2_idx, -1] - iv_surface[T2_idx, 0]

        if abs(skew_T2) < 1e-10 or abs(skew_T1) < 1e-10:
            return 0.1  # fallback

        ratio = (skew_T1 / skew_T2) * (iv_atm_T2 ** 2 / max(iv_atm_T1 ** 2, 1e-10))
        H_hat = -0.5 + np.log(abs(ratio)) / np.log(T1 / T2)
        return float(np.clip(H_hat, 0.01, 0.49))

    def estimate_eta(
        self,
        H_hat: float,
        vix_atmi: float,   # I^VIX_T(0) as T→0
    ) -> float:
        """
        Step 2: Estimate η from VIX ATMI.  (Section 2.2, Alòs & Garcia Lorite 2025)

        lim_{T→0} I^VIX_T(0) = η · √(2H) · Δ^{H-1/2} / (2(H + ½))

        Rearranged:
            η̂ = I^VIX_T(0) · (2Ĥ + 1) / (√(2Ĥ) · Δ^{Ĥ-1/2})

        Args:
            H_hat:    Estimated Hurst parameter.
            vix_atmi: ATM implied volatility of VIX option with short maturity T≈0.

        Returns:
            η̂: Estimated vol-of-vol.
        """
        numerator = vix_atmi * (2 * H_hat + 1)
        denominator = np.sqrt(2 * H_hat) * self.Delta ** (H_hat - 0.5)
        return float(numerator / max(denominator, 1e-10))

    def estimate_rho(
        self,
        H_hat: float,
        eta_hat: float,
        atm_skew: float,   # ∂_k I(T, K*) with T→0: finite difference approximation
        T: float,
    ) -> float:
        """
        Step 3: Estimate ρ from short-time ATMI skew.  (Section 2.2, Alòs et al. 2024)

        lim_{T→0} T^{1/2-H} ∂_K I(0, K*) = 2ηρ√(2H) / (3 + 4H(2+H))

        Rearranged:
            ρ̂ = T^{1/2-Ĥ} · ∂_k I(T, K*) · (3 + 4Ĥ(2+Ĥ)) / (2η̂√(2Ĥ))

        Args:
            H_hat:    Estimated Hurst parameter.
            eta_hat:  Estimated vol-of-vol.
            atm_skew: Finite-difference estimate of ∂_k I(T, K*) at short T.
            T:        Short maturity used for finite difference (T ≈ 0).

        Returns:
            ρ̂: Estimated correlation.
        """
        numerator = T ** (0.5 - H_hat) * atm_skew * (3 + 4 * H_hat * (2 + H_hat))
        denominator = 2 * eta_hat * np.sqrt(2 * H_hat)
        rho_hat = numerator / max(abs(denominator), 1e-10)
        return float(np.clip(rho_hat, -0.999, 0.999))

    def estimate_sigma0(
        self,
        H_hat: float,
        atm_ivs: np.ndarray,   # I(T, K*) for various T
        maturities: np.ndarray,
    ) -> float:
        """
        Step 4: Estimate σ₀ by regression of ATM IV on T^{2H}.  (Section 2.2)

        I(T, K*) ≈ σ₀ + c₁ T^{2H} + O(T^{H+1/2})

        A linear regression of I(T,K*) against T^{2H} gives σ₀ as intercept.

        Args:
            H_hat:      Estimated Hurst parameter.
            atm_ivs:    [nT] ATM implied volatilities.
            maturities: [nT] maturities.

        Returns:
            σ̂₀: Estimated initial volatility.
        """
        T_2H = np.array(maturities) ** (2 * H_hat)
        A = np.column_stack([np.ones_like(T_2H), T_2H])
        coeffs, _, _, _ = np.linalg.lstsq(A, atm_ivs, rcond=None)
        return float(max(coeffs[0], 1e-4))

    def calibrate(
        self,
        iv_surface: np.ndarray,   # [nT, nK]
        maturities: np.ndarray,   # [nT]
        strikes: np.ndarray,      # [nK]
        vix_atmi: float,
        T_short: Optional[float] = None,
        dk: float = 5.0,          # finite difference step for skew estimation
        T1_idx: int = 0,
        T2_idx: int = -1,
    ) -> Dict[str, float]:
        """
        Run the full four-step VIX calibration procedure.  (Section 2.2)

        Args:
            iv_surface: [nT, nK] implied volatility surface.
            maturities: [nT] maturities.
            strikes:    [nK] strikes.
            vix_atmi:   ATM VIX option implied volatility at short maturity.
            T_short:    Short maturity for skew estimation (defaults to maturities[0]).
            dk:         Strike step for finite difference skew approximation.
            T1_idx:     Index of T₁ for H estimation.
            T2_idx:     Index of T₂ for H estimation.

        Returns:
            dict with keys: H, eta, rho, sigma0.
        """
        if T_short is None:
            T_short = float(maturities[0])

        atm_idx = np.argmin(np.abs(strikes - self.S0))

        # Step 1: Estimate H
        H_hat = self.estimate_H(iv_surface, maturities, strikes, T1_idx, T2_idx)
        print(f"[RoughBergomiVIXCalibrator] Step 1: Ĥ = {H_hat:.6f}")

        # Step 2: Estimate η
        eta_hat = self.estimate_eta(H_hat, vix_atmi)
        print(f"[RoughBergomiVIXCalibrator] Step 2: η̂ = {eta_hat:.6f}")

        # Step 3: Estimate ρ from ATM skew (finite difference)
        short_T_idx = np.argmin(np.abs(np.array(maturities) - T_short))
        # ∂_k I ≈ (I(K+dk) - I(K-dk)) / (2dk) at K = S0
        if atm_idx > 0 and atm_idx < len(strikes) - 1:
            atm_skew_fd = (
                iv_surface[short_T_idx, atm_idx + 1]
                - iv_surface[short_T_idx, atm_idx - 1]
            ) / (np.log(strikes[atm_idx + 1]) - np.log(strikes[atm_idx - 1]))
        else:
            atm_skew_fd = 0.0
        rho_hat = self.estimate_rho(H_hat, eta_hat, atm_skew_fd, T_short)
        print(f"[RoughBergomiVIXCalibrator] Step 3: ρ̂ = {rho_hat:.6f}")

        # Step 4: Estimate σ₀
        atm_ivs = iv_surface[:, atm_idx]
        sigma0_hat = self.estimate_sigma0(H_hat, atm_ivs, maturities)
        print(f"[RoughBergomiVIXCalibrator] Step 4: σ̂₀ = {sigma0_hat:.6f}")

        return {
            "H": H_hat,
            "eta": eta_hat,
            "rho": rho_hat,
            "sigma0": sigma0_hat,
        }
