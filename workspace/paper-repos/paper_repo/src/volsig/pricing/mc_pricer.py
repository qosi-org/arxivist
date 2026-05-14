"""
pricing/mc_pricer.py
────────────────────
Monte Carlo European call pricer using the signature-driven price process formula.

Implements Proposition 4.2 (Section 4.1):

    S̃_T(ℓ) = S₀ · exp( ℓᵀ Q(T) ℓ  +  ℓᵀ ∫₀ᵀ vec(S(X)^{≤N}_s) dZ_s )

where:
  - Q(T)_{L(I),L(J)} = -½ <(e_I ⊔ e_J) ⊗ e_0,  S(X)^{≤2N+1}_T>
  - ℓᵀ Q(T) ℓ  is computed via the Cholesky factor U(T): ℓᵀ Q(T) ℓ = -‖U(T)ℓ‖²
"""

from __future__ import annotations

import numpy as np

from volsig.pricing.black_scholes import BlackScholes


class SignatureMCPricer:
    """
    Prices European call options using the signature-driven discounted price
    process of Proposition 4.2.

    Pre-computation:
    ----------------
    The Cholesky factors U[j] and stochastic integrals ∫vec(S)dZ[j] must be
    computed offline (once) and passed at construction.  Given these, pricing
    for any ℓ is a cheap matrix-vector multiply + exponential.

    This separation of offline/online computation is the key to making
    calibration fast.  (Section 4.3, paragraph "Note that signatures are
    computed once (offline) and reused when updating ℓ")
    """

    def __init__(
        self,
        U: np.ndarray,           # [nMC, n_coords, n_coords]  Cholesky of -Q(T)
        stoch_int: np.ndarray,   # [nMC, n_coords]  ∫vec(S^{≤N}_s)dZs per path
        S0: float,
        r: float,
        maturity: float,
    ):
        """
        Args:
            U:         [nMC, n_coords, n_coords] Cholesky factor of -Q(T).
            stoch_int: [nMC, n_coords] stochastic integral ∫₀ᵀ vec(S^{≤N}_s)dZ_s.
            S0:        Initial stock price.
            r:         Risk-free rate.
            maturity:  Maturity T (years).
        """
        self.U = U                  # [nMC, n, n]
        self.stoch_int = stoch_int  # [nMC, n]
        self.S0 = S0
        self.r = r
        self.maturity = maturity
        self.nMC = U.shape[0]
        self.n_coords = U.shape[1]

    def __repr__(self) -> str:
        return (f"SignatureMCPricer(nMC={self.nMC}, n_coords={self.n_coords}, "
                f"S0={self.S0}, T={self.maturity})")

    def simulate_terminal_prices(self, l: np.ndarray) -> np.ndarray:
        """
        Evaluate the discounted terminal price S̃_T(ℓ) for each MC path.

        Proposition 4.2 formula:
            S̃_T(ℓ)(ω_j) = S₀ · exp( -‖U(T)(ω_j) ℓ‖² + ℓᵀ ∫vec(S)dZ(ω_j) )

        Args:
            l: [n_coords] coefficient vector ℓ.

        Returns:
            S_T: [nMC] terminal discounted stock prices.
        """
        assert l.shape == (self.n_coords,), (
            f"ℓ must have shape [{self.n_coords}], got {l.shape}"
        )
        # Quadratic term: -‖U ℓ‖²  (one per path)
        # U: [nMC, n, n], l: [n]  →  U @ l: [nMC, n]
        Ul = self.U @ l                          # [nMC, n_coords]
        quad_term = -np.sum(Ul ** 2, axis=1)     # [nMC]

        # Stochastic integral term: ℓᵀ · stoch_int
        # stoch_int: [nMC, n], l: [n]  →  dot product per path
        linear_term = self.stoch_int @ l          # [nMC]

        exponent = quad_term + linear_term        # [nMC]
        S_T = self.S0 * np.exp(exponent)          # [nMC]
        return S_T

    def price_call(self, l: np.ndarray, K: float) -> float:
        """
        Price a European call with strike K at maturity self.maturity.

        C(K, T, ℓ) = E[(S̃_T(ℓ) - e^{-rT}·K)₊]
                   ≈ (1/nMC) Σ_j max(S̃_T(ℓ)(ω_j) - e^{-rT}·K, 0)

        Args:
            l: [n_coords] coefficient vector ℓ.
            K: Strike price.

        Returns:
            MC estimate of call price.
        """
        S_T = self.simulate_terminal_prices(l)
        disc_K = np.exp(-self.r * self.maturity) * K
        payoffs = np.maximum(S_T - disc_K, 0.0)
        return float(np.mean(payoffs))

    def implied_vol(self, l: np.ndarray, K: float) -> float:
        """
        Compute implied volatility for a given ℓ and strike K.

        Args:
            l: Coefficient vector.
            K: Strike.

        Returns:
            Black-Scholes implied volatility.
        """
        price = self.price_call(l, K)
        return BlackScholes.implied_vol(price, self.S0, K, self.maturity, self.r)


class MultiMaturityPricer:
    """
    Container for SignatureMCPricers across multiple maturities.
    Provides a unified interface for pricing a [nT, nK] surface.
    """

    def __init__(
        self,
        pricers: dict,  # {maturity_float: SignatureMCPricer}
        strikes: np.ndarray,
        S0: float,
        r: float,
    ):
        self.pricers = pricers
        self.strikes = strikes
        self.S0 = S0
        self.r = r
        self.maturities = sorted(pricers.keys())

    def __repr__(self) -> str:
        return f"MultiMaturityPricer(T={self.maturities}, K={list(self.strikes)})"

    def price_surface(self, l: np.ndarray) -> np.ndarray:
        """
        Price the full option surface for coefficient vector ℓ.

        Returns:
            prices: [nT, nK] array of call prices.
        """
        nT = len(self.maturities)
        nK = len(self.strikes)
        prices = np.zeros((nT, nK), dtype=np.float64)
        for i, T in enumerate(self.maturities):
            pricer = self.pricers[T]
            for j, K in enumerate(self.strikes):
                prices[i, j] = pricer.price_call(l, K)
        return prices

    def implied_vol_surface(self, l: np.ndarray) -> np.ndarray:
        """
        Compute implied volatility surface for coefficient vector ℓ.

        Returns:
            ivs: [nT, nK] array of implied volatilities.
        """
        prices = self.price_surface(l)
        return BlackScholes.implied_vol_surface(
            prices, self.S0, self.strikes,
            np.array(self.maturities), self.r
        )


def compute_stochastic_integrals(
    sig_paths: np.ndarray,  # [nMC, T_steps+1, n_coords]
    dZ: np.ndarray,          # [nMC, T_steps]
) -> np.ndarray:
    """
    Compute the stochastic integral ∫₀ᵀ vec(S(X)^{≤N}_s) dZ_s via Euler sum.

    ∫₀ᵀ vec(S^{≤N}_s) dZ_s  ≈  Σ_{t=0}^{T-1} sig_paths[:, t, :] · dZ[:, t]

    Left-point Riemann sum (Itô convention), consistent with the Euler
    simulation of the primary process.

    Args:
        sig_paths: [nMC, T_steps+1, n_coords] signature paths (from SignatureComputer).
        dZ:        [nMC, T_steps] increments of Z = ρW + √(1-ρ²)B.

    Returns:
        stoch_int: [nMC, n_coords] stochastic integrals.
    """
    assert sig_paths.ndim == 3, f"Expected [nMC, T+1, n], got {sig_paths.shape}"
    assert dZ.ndim == 2, f"Expected [nMC, T], got {dZ.shape}"
    nMC, T_plus1, n_coords = sig_paths.shape
    T = T_plus1 - 1
    assert dZ.shape == (nMC, T), (
        f"dZ shape {dZ.shape} inconsistent with sig_paths T_steps={T}"
    )
    # Left-point: sig at time t, multiplied by dZ at step t
    # sig_paths[:, :-1, :] has shape [nMC, T, n_coords]
    # dZ[:, :, None] broadcasts to [nMC, T, 1]
    integrands = sig_paths[:, :-1, :] * dZ[:, :, None]  # [nMC, T, n_coords]
    return np.sum(integrands, axis=1)  # [nMC, n_coords]
