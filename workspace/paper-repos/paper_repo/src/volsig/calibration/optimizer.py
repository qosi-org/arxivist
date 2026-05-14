"""
calibration/optimizer.py
────────────────────────
L-BFGS-B optimizer for the signature calibration loss function.

Implements the calibration procedure of Section 4.2 / Algorithm (Section 4.3):

    L(ℓ) = Σᵢ γᵢ (C_mkt(Kᵢ,Tᵢ) − C(Kᵢ,Tᵢ,ℓ))²

where:
  - C_mkt are synthetic market option prices
  - C(K,T,ℓ) are MC prices from Proposition 4.2
  - γᵢ = 1/Vega_i are inverse-Vega weights  (Section 4.2)
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from volsig.pricing.mc_pricer import MultiMaturityPricer
from volsig.pricing.black_scholes import BlackScholes


class SignatureCalibrator:
    """
    Calibrates the coefficient vector ℓ ∈ ℝ^{n_coords} by minimising the
    weighted least-squares loss between market and model option prices.

    Paper: Section 4.2 (loss function), Section 4.3 (algorithm, L-BFGS-B).
    """

    def __init__(
        self,
        market_prices: np.ndarray,    # [nT, nK]  flattened in C order
        strikes: np.ndarray,           # [nK]
        maturities: np.ndarray,        # [nT]
        pricer: MultiMaturityPricer,
        S0: float,
        r: float,
        sigma0: float,
        weight_scheme: str = "inverse_vega",
        box_bounds: Tuple[float, float] = (-10.0, 10.0),  # ASSUMED; see SIR ambiguity R5
        max_iter: int = 10_000,        # ASSUMED
        tol: float = 1e-8,             # paper: Section 4.3
    ):
        """
        Args:
            market_prices:  [nT, nK] synthetic market call prices.
            strikes:        [nK] strike grid.
            maturities:     [nT] maturity grid.
            pricer:         MultiMaturityPricer (precomputed offline).
            S0:             Initial stock price.
            r:              Risk-free rate.
            sigma0:         Initial volatility (for inverse-Vega weight computation).
            weight_scheme:  'inverse_vega' (paper) or 'uniform'.
            box_bounds:     (lower, upper) box bounds for ℓ components.
            max_iter:       Maximum L-BFGS-B iterations.
            tol:            Convergence tolerance.
        """
        self.market_prices = market_prices.flatten()    # [nT*nK]
        self.strikes = strikes
        self.maturities = maturities
        self.pricer = pricer
        self.S0 = S0
        self.r = r
        self.tol = tol
        self.max_iter = max_iter
        self.box_bounds_val = box_bounds

        nT, nK = market_prices.shape
        self.nT = nT
        self.nK = nK
        self.n_contracts = nT * nK

        # Compute weights
        self.weights = self._compute_weights(weight_scheme, sigma0)
        self._call_count = 0
        self._loss_history: List[float] = []

    def _compute_weights(self, scheme: str, sigma0: float) -> np.ndarray:
        """
        Compute calibration weights γᵢ.

        Paper (Section 4.2): γᵢ = 1/Vega_i, where Vega is evaluated under
        Black-Scholes with σ₀.
        """
        if scheme == "inverse_vega":
            w = BlackScholes.inverse_vega_weights(
                self.S0, self.strikes, self.maturities, self.r, sigma0
            )
        elif scheme == "uniform":
            w = np.ones(self.n_contracts)
        else:
            raise ValueError(f"Unknown weight_scheme '{scheme}'")
        # Normalise so weights sum to n_contracts (preserves loss scale)
        w = w / w.mean()
        return w

    def loss(self, l: np.ndarray) -> float:
        """
        Evaluate the calibration loss function L(ℓ).  (Section 4.2, Eq. 4.7)

        L(ℓ) = Σᵢ γᵢ (C_mkt(Kᵢ,Tᵢ) − C(Kᵢ,Tᵢ,ℓ))²

        Args:
            l: [n_coords] coefficient vector.

        Returns:
            Scalar loss value.
        """
        model_prices = self.pricer.price_surface(l).flatten()  # [nT*nK]
        residuals = self.market_prices - model_prices           # [nT*nK]
        l_val = float(np.sum(self.weights * residuals ** 2))

        self._call_count += 1
        self._loss_history.append(l_val)
        if self._call_count % 50 == 0:
            print(f"  [iter {self._call_count:5d}]  L(ℓ) = {l_val:.6e}")
        return l_val

    def calibrate(
        self,
        l0: Optional[np.ndarray] = None,
    ) -> OptimizeResult:
        """
        Run L-BFGS-B optimisation to find ℓ* minimising L(ℓ).

        Args:
            l0: Initial coefficient vector. If None, initialise to zeros.
                # ASSUMED: paper does not specify initialisation.

        Returns:
            scipy.optimize.OptimizeResult with field .x = ℓ*.
        """
        n_coords = self.pricer.pricers[self.maturities[0]].n_coords

        if l0 is None:
            l0 = np.zeros(n_coords, dtype=np.float64)  # ASSUMED

        lo, hi = self.box_bounds_val
        bounds = [(lo, hi)] * n_coords  # ASSUMED bounds

        print(f"[SignatureCalibrator] Starting L-BFGS-B optimisation")
        print(f"  n_coords={n_coords}, n_contracts={self.n_contracts}, "
              f"bounds=[{lo},{hi}], tol={self.tol}")
        t0 = time.time()

        result = minimize(
            fun=self.loss,
            x0=l0,
            method="L-BFGS-B",       # paper: Section 4.3
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": 1e-9},
        )

        elapsed = time.time() - t0
        print(f"[SignatureCalibrator] Done in {elapsed:.1f}s  "
              f"({self._call_count} loss evaluations)")
        print(f"  Final loss: {result.fun:.6e}  |  success: {result.success}")
        print(f"  Message: {result.message}")

        # Check if bounds are active (diagnostic for Risk R5)
        lo_arr = np.full(n_coords, lo)
        hi_arr = np.full(n_coords, hi)
        n_active_lo = np.sum(result.x <= lo_arr + 1e-6)
        n_active_hi = np.sum(result.x >= hi_arr - 1e-6)
        if n_active_lo > 0 or n_active_hi > 0:
            print(f"  WARNING: {n_active_lo} components at lower bound, "
                  f"{n_active_hi} at upper bound. "
                  "Consider widening box_bounds in config.")

        return result

    def compute_iv_errors(
        self,
        l_star: np.ndarray,
        iv_market: np.ndarray,   # [nT, nK]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute absolute IV errors |IV_SIG(Ki,Ti) - IV_mkt(Ki,Ti)|.
        Replicates the error tables in Tables 5.1, 5.2, 6.1 of the paper.

        Args:
            l_star:    Calibrated ℓ* vector.
            iv_market: [nT, nK] ground-truth implied volatility surface.

        Returns:
            iv_model: [nT, nK] model-implied volatility surface.
            errors:   [nT, nK] absolute errors.
        """
        iv_model = self.pricer.implied_vol_surface(l_star)
        errors = np.abs(iv_model - iv_market)
        return iv_model, errors
