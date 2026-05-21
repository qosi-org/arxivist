"""
monte_carlo/mc_pricer.py
========================
Reference Monte Carlo pricer for barrier options under correlated
stochastic-clock models (rho != 0).

Used to validate the semi-analytic formulas of Sections 2-5.

Simulation procedure (Section 6.1):
  1. Simulate variance path v_t (CIR full-truncation Euler or exact OU)
  2. Accumulate Gamma_t by trapezoidal integration
  3. Simulate correlated log-price X_t via orthogonal decomposition
  4. Apply Brownian-bridge continuity correction for barrier monitoring
     (Broadie, Glasserman & Kou, 1997)

Paper: Section 6.1, Eqs (6.1)-(6.6).
arXiv: 2605.06677v1
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal
from ..models.base_clock import BaseClock
from ..models.cir_clock import CIRClock
from ..models.sq_ou_clock import SquaredOUClock


ClockFamily = Literal["cir", "sq_ou"]


@dataclass
class MCResult:
    """Result of a Monte Carlo pricing run."""
    price: float
    stderr: float
    n_paths: int
    n_steps: int
    rho: float
    barrier_type: str
    clock_family: str

    def __str__(self) -> str:
        return (
            f"MC Price: {self.price:.6f} ± {self.stderr:.6f} "
            f"(95% CI: [{self.price-1.96*self.stderr:.6f}, "
            f"{self.price+1.96*self.stderr:.6f}])"
        )


class MonteCarloPricer:
    """Reference Monte Carlo pricer for correlated stochastic-clock barrier options.

    Supports CIR and Squared-OU clocks with arbitrary correlation rho.
    Includes Brownian-bridge continuity correction for discrete barrier monitoring.

    Paper: Section 6.1, Eqs (6.1)-(6.6).
    """

    def __init__(
        self,
        clock: BaseClock,
        n_paths: int = 1_000_000,
        n_steps_per_year: int = 2080,
        bridge_correction: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        clock              : BaseClock instance (CIR or SquaredOU)
        n_paths            : Number of simulation paths (paper uses 10^6)
        n_steps_per_year   : Steps per year (paper uses 2080 = 8/day * 260 days)
        bridge_correction  : Apply Brownian-bridge correction (Broadie et al. 1997)
        seed               : Random seed for reproducibility
        """
        self.clock = clock
        self.n_paths = n_paths
        self.n_steps_per_year = n_steps_per_year
        self.bridge_correction = bridge_correction
        self.seed = seed

    # ------------------------------------------------------------------
    # Correlated path simulation  (Section 6.1, Eq 6.6)
    # ------------------------------------------------------------------

    def _simulate_correlated(
        self,
        T: float,
        rho: float,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate correlated (X, v, Gamma) paths.

        Correlated Brownian drivers via orthogonal decomposition (Eq 6.6):
            W^(v) = W_tilde^(v)
            W^(S) = rho * W_tilde^(v) + sqrt(1-rho²) * W_tilde^(S)
        with W_tilde^(v), W_tilde^(S) independent.

        Returns
        -------
        x_paths     : (n_paths, n_steps+1)  log-forward price paths
        v_paths     : (n_paths, n_steps+1)  variance paths
        gamma_paths : (n_paths, n_steps+1)  clock paths
        """
        dt = T / n_steps
        beta = -0.5                          # Ito drift

        x_paths     = np.empty((n_paths, n_steps + 1))
        v_paths     = np.empty((n_paths, n_steps + 1))
        gamma_paths = np.empty((n_paths, n_steps + 1))

        # Initial conditions
        x_paths[:, 0]     = np.log(self._get_F0())  # placeholder; set by price()
        gamma_paths[:, 0] = 0.0

        if isinstance(self.clock, CIRClock):
            v_paths[:, 0] = self.clock.v0
            sqrt_dt = np.sqrt(dt)

            for i in range(n_steps):
                # Independent Brownian increments
                Z_v = rng.standard_normal(n_paths)
                Z_s = rng.standard_normal(n_paths)
                # Correlated increment for X driver (Eq 6.6)
                Z_x = rho * Z_v + np.sqrt(1.0 - rho**2) * Z_s

                v_pos = np.maximum(v_paths[:, i], 0.0)

                # CIR variance step (full-truncation Euler, Eq 6.2)
                v_paths[:, i + 1] = (
                    v_paths[:, i]
                    + self.clock.kappa * (self.clock.theta - v_pos) * dt
                    + self.clock.xi * np.sqrt(v_pos) * sqrt_dt * Z_v
                )

                # Clock: trapezoidal integration (Eq 6.3)
                gamma_paths[:, i + 1] = (
                    gamma_paths[:, i]
                    + 0.5 * (v_paths[:, i] + v_paths[:, i + 1]) * dt
                )

                # Log-price: dX = beta*v*dt + sqrt(v)*dW^(S)
                v_eff = np.maximum(v_paths[:, i], 0.0)
                x_paths[:, i + 1] = (
                    x_paths[:, i]
                    + beta * v_eff * dt
                    + np.sqrt(v_eff * dt) * Z_x
                )

        elif isinstance(self.clock, SquaredOUClock):
            v_paths[:, 0] = self.clock.v0    # = Y0²
            nu_paths = np.empty((n_paths, n_steps + 1))
            nu_paths[:, 0] = self.clock.Y0

            exp_adt = np.exp(-self.clock.alpha * dt)
            cond_std = self.clock.sigma * np.sqrt(
                (1.0 - np.exp(-2.0 * self.clock.alpha * dt))
                / (2.0 * self.clock.alpha)
            )

            for i in range(n_steps):
                Z_v = rng.standard_normal(n_paths)
                Z_s = rng.standard_normal(n_paths)
                Z_x = rho * Z_v + np.sqrt(1.0 - rho**2) * Z_s

                # Exact OU step (Eq 6.5)
                nu_paths[:, i + 1] = exp_adt * nu_paths[:, i] + cond_std * Z_v
                v_paths[:, i + 1]  = nu_paths[:, i + 1] ** 2

                gamma_paths[:, i + 1] = (
                    gamma_paths[:, i]
                    + 0.5 * (v_paths[:, i] + v_paths[:, i + 1]) * dt
                )

                v_eff = v_paths[:, i]
                x_paths[:, i + 1] = (
                    x_paths[:, i]
                    + beta * v_eff * dt
                    + np.sqrt(np.maximum(v_eff * dt, 0.0)) * Z_x
                )
        else:
            raise NotImplementedError(
                f"MC simulation not implemented for {type(self.clock).__name__}"
            )

        return x_paths, v_paths, gamma_paths

    def _get_F0(self) -> float:
        return getattr(self, '_F0', 100.0)

    # ------------------------------------------------------------------
    # Brownian-bridge barrier correction  (Broadie et al. 1997)
    # ------------------------------------------------------------------

    @staticmethod
    def bridge_survival_prob(
        x_a: np.ndarray,
        x_b: np.ndarray,
        barrier: float,
        dt: float,
        v: np.ndarray,
        barrier_side: str = "upper",
    ) -> np.ndarray:
        """Probability of NOT hitting barrier between steps a and b.

        Broadie-Glasserman-Kou continuity correction (1997):
            P(M_{a,b} < h | X_a, X_b) = 1 - exp(-2*(h-X_a)*(h-X_b) / (v*dt))
        for an upper barrier h, where M_{a,b} = sup_{a<=t<=b} X_t.

        Paper: Section 6.1, step (v) — 'Brownian-bridge continuity correction'.
        """
        if barrier_side == "upper":
            # Both must be below barrier
            above_a = x_a >= barrier
            above_b = x_b >= barrier
            already_hit = above_a | above_b

            delta_a = barrier - x_a
            delta_b = barrier - x_b
            v_dt = np.maximum(v * dt, 1e-14)
            p_not_hit = np.where(
                already_hit,
                0.0,
                np.exp(-2.0 * delta_a * delta_b / v_dt)
            )
        else:  # lower barrier
            delta_a = x_a - barrier
            delta_b = x_b - barrier
            v_dt = np.maximum(v * dt, 1e-14)
            below_a = x_a <= barrier
            below_b = x_b <= barrier
            already_hit = below_a | below_b
            p_not_hit = np.where(
                already_hit,
                0.0,
                np.exp(-2.0 * delta_a * delta_b / v_dt)
            )
        return p_not_hit

    # ------------------------------------------------------------------
    # Main pricing method
    # ------------------------------------------------------------------

    def price(
        self,
        F0: float,
        K: float,
        T: float,
        r: float = 0.0,
        q: float = 0.0,
        rho: float = 0.0,
        L: float = None,
        H: float = None,
        barrier_type: str = "doc",
        n_paths: int = None,
        n_steps: int = None,
        seed: int = None,
    ) -> MCResult:
        """Price a single- or double-barrier option by Monte Carlo.

        Parameters
        ----------
        F0, K       : float  Forward and strike prices
        T           : float  Maturity
        r, q        : float  Risk-free rate and dividend yield
        rho         : float  Return-volatility correlation
        L, H        : float  Lower/upper barrier levels (set as needed)
        barrier_type: str    'uop' | 'doc' | 'dko_call' | 'dko_put'
        n_paths     : int    Override self.n_paths
        n_steps     : int    Override computed steps
        seed        : int    Override self.seed

        Returns
        -------
        MCResult dataclass
        """
        n_paths = n_paths or self.n_paths
        T_steps = int(round(T * self.n_steps_per_year))
        n_steps = n_steps or T_steps
        rng     = np.random.default_rng(seed or self.seed)

        self._F0 = F0
        discount = np.exp(-r * T)
        dt = T / n_steps
        x0 = np.log(F0)

        x_paths, v_paths, gamma_paths = self._simulate_correlated(
            T, rho, n_paths, n_steps, rng
        )
        x_paths[:, 0] = x0   # correct initial condition

        # ------ Barrier monitoring ------
        survived = np.ones(n_paths, dtype=bool)
        # Track survival probability per path (for bridge correction)
        log_surv_prob = np.zeros(n_paths)

        log_L = np.log(L) if L is not None else -np.inf
        log_H = np.log(H) if H is not None else  np.inf

        for i in range(n_steps):
            x_a = x_paths[:, i]
            x_b = x_paths[:, i + 1]
            v_i = np.maximum(v_paths[:, i], 0.0)

            if H is not None:
                # Hard kill
                hard_hit_up = (x_b >= log_H) | (x_a >= log_H)
                survived &= ~hard_hit_up

                if self.bridge_correction:
                    p_cross = self.bridge_survival_prob(
                        x_a, x_b, log_H, dt, v_i, "upper"
                    )
                    log_surv_prob = np.where(survived, log_surv_prob + np.log(np.maximum(1.0 - p_cross, 1e-14)), log_surv_prob)

            if L is not None:
                hard_hit_dn = (x_b <= log_L) | (x_a <= log_L)
                survived &= ~hard_hit_dn

                if self.bridge_correction:
                    p_cross = self.bridge_survival_prob(
                        x_a, x_b, log_L, dt, v_i, "lower"
                    )
                    log_surv_prob = np.where(survived, log_surv_prob + np.log(np.maximum(1.0 - p_cross, 1e-14)), log_surv_prob)

        # ------ Terminal payoff ------
        S_T = np.exp(x_paths[:, -1])
        if barrier_type in ("uop",):
            payoff = np.maximum(K - S_T, 0.0)
        elif barrier_type in ("doc", "dko_call"):
            payoff = np.maximum(S_T - K, 0.0)
        elif barrier_type in ("dko_put",):
            payoff = np.maximum(K - S_T, 0.0)
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

        # Apply survival and bridge correction
        if self.bridge_correction:
            surv_weight = np.where(survived, np.exp(log_surv_prob), 0.0)
            discounted = discount * payoff * surv_weight
        else:
            discounted = discount * payoff * survived.astype(float)

        price  = float(np.mean(discounted))
        stderr = float(np.std(discounted) / np.sqrt(n_paths))

        return MCResult(
            price=price,
            stderr=stderr,
            n_paths=n_paths,
            n_steps=n_steps,
            rho=rho,
            barrier_type=barrier_type,
            clock_family=type(self.clock).__name__,
        )

    def __repr__(self) -> str:
        return (
            f"MonteCarloPricer(clock={self.clock!r}, "
            f"n_paths={self.n_paths}, n_steps_per_year={self.n_steps_per_year})"
        )
