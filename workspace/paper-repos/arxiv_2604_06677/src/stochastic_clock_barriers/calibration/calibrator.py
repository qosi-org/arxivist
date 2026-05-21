"""
calibration/calibrator.py
=========================
4-stage calibration workflow for stochastic-clock barrier models.

Stage 1: Fit clock parameters theta to vanilla implied-vol surface
         using characteristic-function pricing (Section 7.2).
Stage 2: Refine with small barrier/no-touch set at rho=0
         to pin down path-sensitive degrees of freedom (Section 7.3).
Stage 3: Leverage calibration via cached rho-expansion coefficients
         (fast polynomial/Padé scan, Section 7.4 Stage 3).
Stage 4: Joint refinement with regularization (Section 7.4 Stage 4).

Initialization from variance swaps and ATM volatility (Section 7.5).

Paper: Section 7, Eqs (7.4)-(7.11).
arXiv: 2605.06677v1
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
from typing import Optional
from ..models.cir_clock import CIRClock
from ..models.sq_ou_clock import SquaredOUClock
from ..pricing.single_barrier import SingleBarrierPricer
from ..pricing.double_barrier import DoubleBarrierPricer
from ..pricing.vanilla import VanillaPricer
from ..pricing.leverage_expansion import LeverageExpansion
from ..pricing.pade_accelerator import PadeAccelerator


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VanillaQuote:
    """A single vanilla implied-volatility market quote."""
    F0: float
    K: float
    T: float
    r: float
    q: float
    market_iv: float       # Black-Scholes implied vol
    weight: float = 1.0    # Inverse bid-ask variance proxy


@dataclass
class BarrierQuote:
    """A single barrier option market quote (price or IV)."""
    F0: float
    K: float
    T: float
    r: float
    q: float
    market_price: float
    barrier_type: str      # 'uop' | 'doc' | 'dko_call' | 'dko_put'
    L: Optional[float] = None
    H: Optional[float] = None
    weight: float = 1.0


@dataclass
class CalibrationResult:
    """Output of the 4-stage calibration."""
    clock_params: dict
    rho: float
    vanilla_rmse: float
    barrier_rmse: float
    n_iterations: int
    converged: bool
    diagnostics: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "CalibrationResult:",
            f"  Clock params : {self.clock_params}",
            f"  rho          : {self.rho:.4f}",
            f"  Vanilla RMSE : {self.vanilla_rmse:.6f}",
            f"  Barrier RMSE : {self.barrier_rmse:.6f}",
            f"  Converged    : {self.converged}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class Calibrator:
    """4-stage calibration engine for stochastic-clock barrier models.

    Supports CIR and Squared-OU clocks.

    Paper: Section 7.
    """

    def __init__(
        self,
        clock_family: str = "cir",
        vanilla_quotes: list[VanillaQuote] = None,
        barrier_quotes: list[BarrierQuote] = None,
        vanilla_weight: float = 1.0,
        barrier_weight: float = 0.5,
        regularization_lambda: float = 1e-3,
        expansion_order: int = 3,
        optimizer: str = "L-BFGS-B",
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        clock_family           : 'cir' or 'sq_ou'
        vanilla_quotes         : list of VanillaQuote
        barrier_quotes         : list of BarrierQuote
        vanilla_weight         : Weight on vanilla errors in joint objective (Eq 7.4)
        barrier_weight         : Weight on barrier errors (typically 0.3-0.5)
        regularization_lambda  : L2 regularization strength (Section 7.2.3)
        expansion_order        : rho-expansion order for Stage 3
        optimizer              : scipy optimizer name
        verbose                : Print progress
        """
        self.clock_family = clock_family
        self.vanilla_quotes = vanilla_quotes or []
        self.barrier_quotes = barrier_quotes or []
        self.vanilla_weight = vanilla_weight
        self.barrier_weight = barrier_weight
        self.reg_lambda = regularization_lambda
        self.expansion_order = expansion_order
        self.optimizer = optimizer
        self.verbose = verbose

        # Cached expansion coefficients per barrier instrument (Stage 3)
        self._cached_coefficients: dict[int, list[float]] = {}
        self._current_clock: Optional[CIRClock | SquaredOUClock] = None

    # ------------------------------------------------------------------
    # Clock construction from parameter vector
    # ------------------------------------------------------------------

    def _build_clock(self, params: np.ndarray) -> CIRClock | SquaredOUClock:
        """Construct clock from unconstrained parameter vector.

        Uses exponential reparameterization to enforce positivity (Section 7.5.4).
        For CIR: params = [log(kappa), log(theta), log(xi), log(v0)]
        For SquaredOU: params = [log(alpha), log(sigma), Y0]
        """
        if self.clock_family == "cir":
            kappa, theta, xi, v0 = np.exp(params[:4])
            return CIRClock(kappa=kappa, theta=theta, xi=xi, v0=v0)
        elif self.clock_family == "sq_ou":
            alpha, sigma = np.exp(params[:2])
            Y0 = params[2]        # Y0 can be negative; v0 = Y0²
            return SquaredOUClock(alpha=alpha, sigma=sigma, Y0=Y0)
        else:
            raise ValueError(f"Unknown clock_family: {self.clock_family}")

    def _clock_to_params(self, clock) -> np.ndarray:
        """Extract unconstrained parameter vector from clock."""
        if isinstance(clock, CIRClock):
            return np.log([clock.kappa, clock.theta, clock.xi, clock.v0])
        elif isinstance(clock, SquaredOUClock):
            return np.array([np.log(clock.alpha), np.log(clock.sigma), clock.Y0])

    # ------------------------------------------------------------------
    # Initialization  (Section 7.5)
    # ------------------------------------------------------------------

    def initialize_from_variance_swaps(
        self,
        var_swap_strikes: dict[float, float],
        atm_iv: Optional[float] = None,
    ) -> dict:
        """Initialize clock parameters from variance swap term structure.

        Paper: Section 7.5.1, Eq (7.7):
            K_var(T) ≈ theta + (v0 - theta) * (1 - exp(-kappa*T)) / (kappa*T)

        Parameters
        ----------
        var_swap_strikes : {T: K_var(T)} — realized variance swap strikes
        atm_iv           : ATM implied vol for dispersion initialization

        Returns
        -------
        dict with initialized parameters
        """
        maturities = np.array(sorted(var_swap_strikes.keys()))
        kvar_vals  = np.array([var_swap_strikes[T] for T in maturities])

        if self.clock_family == "cir":
            # Step (i): v0 from shortest maturity
            v0_init = kvar_vals[0] * 0.9

            # Step (ii): theta from long end
            theta_init = kvar_vals[-1]

            # Step (iii): kappa from LS fit of Eq (7.7)
            from scipy.optimize import curve_fit

            def kvar_model(T, kappa):
                return theta_init + (v0_init - theta_init) * (1 - np.exp(-kappa * T)) / (kappa * T)

            try:
                kappa_opt, _ = curve_fit(kvar_model, maturities, kvar_vals, p0=[1.0], bounds=(0.01, 20.0))
                kappa_init = float(kappa_opt[0])
            except Exception:
                kappa_init = 1.0

            # Dispersion from ATM convexity (Section 7.5.2)
            xi_init = 0.4 if atm_iv is None else max(0.1, atm_iv * 2.0)

            params = {"kappa": kappa_init, "theta": theta_init,
                      "xi": xi_init, "v0": v0_init}

        elif self.clock_family == "sq_ou":
            v0_init    = kvar_vals[0]
            theta_init = kvar_vals[-1]
            alpha_init = 1.0
            sigma_init = 0.4 if atm_iv is None else atm_iv
            Y0_init    = np.sqrt(v0_init)
            params = {"alpha": alpha_init, "sigma": sigma_init, "Y0": Y0_init}

        if self.verbose:
            print(f"[Calibration] Initialized from variance swaps: {params}")
        return params

    # ------------------------------------------------------------------
    # Objective: vanilla errors  (Section 7.2.2, Eq 7.4)
    # ------------------------------------------------------------------

    def _vanilla_objective(self, params: np.ndarray) -> float:
        """Weighted sum of squared IV errors on vanilla surface."""
        try:
            clock = self._build_clock(params)
        except Exception:
            return 1e6

        pricer = VanillaPricer(clock)
        obj = 0.0
        for q in self.vanilla_quotes:
            try:
                model_iv = pricer.implied_vol(q.F0, q.K, q.T, q.r)
                err = (model_iv - q.market_iv) * q.weight
                obj += err**2
            except Exception:
                obj += 1.0   # penalty for failed evaluation

        # L2 regularization (Section 7.2.3)
        obj += self.reg_lambda * np.sum(params**2)
        return float(obj)

    # ------------------------------------------------------------------
    # Objective: barrier errors at rho=0  (Stage 2)
    # ------------------------------------------------------------------

    def _barrier_objective_rho0(self, params: np.ndarray) -> float:
        """Weighted squared errors on barrier quotes at rho=0."""
        try:
            clock = self._build_clock(params)
        except Exception:
            return 1e6

        sb = SingleBarrierPricer(clock)
        db = DoubleBarrierPricer(clock)
        obj = 0.0

        for q in self.barrier_quotes:
            try:
                if q.barrier_type == "uop":
                    model_p = sb.price_uop(q.F0, q.K, q.H, q.T, q.r, q.q)
                elif q.barrier_type == "doc":
                    model_p = sb.price_doc(q.F0, q.K, q.L, q.T, q.r, q.q)
                elif q.barrier_type == "dko_call":
                    model_p = db.price_dko_call(q.F0, q.K, q.L, q.H, q.T, q.r, q.q)
                elif q.barrier_type == "dko_put":
                    model_p = db.price_dko_put(q.F0, q.K, q.L, q.H, q.T, q.r, q.q)
                else:
                    continue
                err = (model_p - q.market_price) * q.weight
                obj += err**2
            except Exception:
                obj += 1.0
        return float(obj)

    # ------------------------------------------------------------------
    # Stage 3: leverage calibration via cached expansion  (Section 7.4)
    # ------------------------------------------------------------------

    def _cache_expansion_coefficients(
        self, clock: CIRClock | SquaredOUClock
    ) -> None:
        """Precompute rho-expansion coefficients for all barrier quotes.

        Paper: Section 7.4 Stage 3 — 'cache the expansion coefficients
        {C_n(theta)} then scan over rho at negligible marginal cost.'
        """
        if self.verbose:
            print("[Calibration Stage 3] Caching rho-expansion coefficients...")

        self._cached_coefficients = {}
        for i, q in enumerate(self.barrier_quotes):
            contract = {
                "F0": q.F0, "K": q.K, "T": q.T,
                "r": q.r, "q": q.q,
            }
            if q.L is not None:
                contract["L"] = q.L
            if q.H is not None:
                contract["H"] = q.H

            exp = LeverageExpansion(
                clock, contract, barrier_type=q.barrier_type,
                n_paths=50_000, n_steps=50, seed=42,
            )
            try:
                coeffs = exp.compute_coefficients(order=self.expansion_order)
                self._cached_coefficients[i] = coeffs
                if self.verbose:
                    print(f"  Barrier {i} ({q.barrier_type}): C = {[f'{c:.4f}' for c in coeffs]}")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: expansion failed for barrier {i}: {e}")
                self._cached_coefficients[i] = [0.0]

    def _rho_objective(self, rho: float) -> float:
        """Cheap rho scan using cached polynomial evaluation (Eq 7.5).

        Paper: Section 7.4 Stage 3, Eq (7.5):
            Q^mod(theta, rho) ≈ sum_n C_n(theta) * rho^n
        """
        obj = 0.0
        for i, q in enumerate(self.barrier_quotes):
            coeffs = self._cached_coefficients.get(i, [0.0])
            try:
                pade = PadeAccelerator(coeffs)
                if len(coeffs) >= 5:
                    pade.build(2, 2)
                    model_p, _ = pade.evaluate_safe(rho)
                else:
                    model_p = sum(c * rho**n for n, c in enumerate(coeffs))
            except Exception:
                model_p = coeffs[0] if coeffs else 0.0
            err = (model_p - q.market_price) * q.weight
            obj += err**2
        return float(obj)

    # ------------------------------------------------------------------
    # Stage 4: joint objective  (Section 7.4 Stage 4)
    # ------------------------------------------------------------------

    def _joint_objective(self, all_params: np.ndarray) -> float:
        """Joint vanilla + barrier objective with regularization (Eq 7.4)."""
        clock_params = all_params[:-1]
        rho = float(np.clip(all_params[-1], -0.99, 0.99))

        van_obj = self._vanilla_objective(clock_params)

        try:
            clock = self._build_clock(clock_params)
            sb = SingleBarrierPricer(clock)
            db = DoubleBarrierPricer(clock)
            bar_obj = 0.0
            for q in self.barrier_quotes:
                coeffs = self._cached_coefficients.get(
                    self.barrier_quotes.index(q), None
                )
                if coeffs:
                    model_p = sum(c * rho**n for n, c in enumerate(coeffs))
                else:
                    model_p = 0.0
                err = (model_p - q.market_price) * q.weight
                bar_obj += err**2
        except Exception:
            bar_obj = 1e4

        return (
            self.vanilla_weight * van_obj
            + self.barrier_weight * bar_obj
            + self.reg_lambda * np.sum(clock_params**2)
        )

    # ------------------------------------------------------------------
    # Public: run full 4-stage calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        init_params: Optional[dict] = None,
        init_rho: float = -0.3,
    ) -> CalibrationResult:
        """Execute the 4-stage calibration workflow.

        Paper: Section 7.4.

        Parameters
        ----------
        init_params : Optional[dict]  Initial clock parameter dict.
                      If None, uses heuristic defaults.
        init_rho    : float  Initial guess for leverage correlation.
        """
        # ------ Default initialization ------
        if init_params is None:
            if self.clock_family == "cir":
                init_params = {"kappa": 1.0, "theta": 0.04, "xi": 0.4, "v0": 0.04}
            else:
                init_params = {"alpha": 1.0, "sigma": 0.4, "Y0": 0.2}

        if self.clock_family == "cir":
            x0 = np.log([init_params["kappa"], init_params["theta"],
                         init_params["xi"], init_params["v0"]])
            bounds = [(-4, 4), (-6, 2), (-4, 2), (-6, 0)]   # log-space bounds
        else:
            x0 = np.array([np.log(init_params["alpha"]),
                           np.log(init_params["sigma"]),
                           init_params["Y0"]])
            bounds = [(-3, 3), (-4, 2), (-3, 3)]

        # ------ Stage 1: vanilla fit ------
        if self.verbose:
            print("[Calibration Stage 1] Fitting clock parameters to vanillas...")

        if self.vanilla_quotes:
            res1 = minimize(
                self._vanilla_objective, x0,
                method=self.optimizer,
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            x0 = res1.x
            vanilla_rmse = np.sqrt(res1.fun / max(len(self.vanilla_quotes), 1))
            if self.verbose:
                print(f"  Stage 1 RMSE: {vanilla_rmse:.6f}")
        else:
            vanilla_rmse = 0.0
            if self.verbose:
                print("  No vanilla quotes — skipping Stage 1.")

        clock_stage1 = self._build_clock(x0)

        # ------ Stage 2: barrier refinement at rho=0 ------
        if self.verbose:
            print("[Calibration Stage 2] Barrier refinement at rho=0...")

        if self.barrier_quotes:
            def combined_stage2(p):
                return (self.vanilla_weight * self._vanilla_objective(p)
                        + self.barrier_weight * self._barrier_objective_rho0(p))

            res2 = minimize(
                combined_stage2, x0,
                method=self.optimizer,
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-9},
            )
            x0 = res2.x
            if self.verbose:
                print(f"  Stage 2 loss: {res2.fun:.6f}")

        clock_stage2 = self._build_clock(x0)

        # ------ Stage 3: leverage calibration via cached expansion ------
        if self.verbose:
            print("[Calibration Stage 3] Leverage rho calibration...")

        rho_opt = init_rho
        if self.barrier_quotes:
            self._cache_expansion_coefficients(clock_stage2)

            # Scan rho over [-0.99, 0.99]
            from scipy.optimize import minimize_scalar
            res3 = minimize_scalar(
                self._rho_objective,
                bounds=(-0.99, 0.99),
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 100},
            )
            rho_opt = float(res3.x)
            if self.verbose:
                print(f"  Stage 3 rho = {rho_opt:.4f}")

        # ------ Stage 4: joint refinement ------
        if self.verbose:
            print("[Calibration Stage 4] Joint refinement...")

        all_params = np.append(x0, rho_opt)
        all_bounds = bounds + [(-0.99, 0.99)]

        res4 = minimize(
            self._joint_objective, all_params,
            method=self.optimizer,
            bounds=all_bounds,
            options={"maxiter": 200, "ftol": 1e-9},
        )

        final_clock_params = self._build_clock(res4.x[:-1])
        final_rho = float(np.clip(res4.x[-1], -0.99, 0.99))

        # ------ Compute final errors ------
        bar_rmse = np.sqrt(self._barrier_objective_rho0(res4.x[:-1])
                           / max(len(self.barrier_quotes), 1))
        van_rmse = np.sqrt(self._vanilla_objective(res4.x[:-1])
                           / max(len(self.vanilla_quotes), 1))

        if isinstance(final_clock_params, CIRClock):
            clock_dict = {
                "kappa": final_clock_params.kappa,
                "theta": final_clock_params.theta,
                "xi":    final_clock_params.xi,
                "v0":    final_clock_params.v0,
            }
        else:
            clock_dict = {
                "alpha": final_clock_params.alpha,
                "sigma": final_clock_params.sigma,
                "Y0":    final_clock_params.Y0,
            }

        result = CalibrationResult(
            clock_params=clock_dict,
            rho=final_rho,
            vanilla_rmse=van_rmse,
            barrier_rmse=bar_rmse,
            n_iterations=res4.nit,
            converged=res4.success,
            diagnostics={
                "final_objective": res4.fun,
                "stage4_message": res4.message,
            },
        )

        if self.verbose:
            print(f"\n{result}")

        return result
