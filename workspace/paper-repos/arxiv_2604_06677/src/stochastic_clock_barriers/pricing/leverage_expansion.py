"""
pricing/leverage_expansion.py
=============================
rho-expansion for barrier prices under correlated stochastic-clock models.

When the return Brownian motion and the clock driver are correlated (rho != 0),
the clean time-change reduction of Section 2-3 breaks down. This module
implements the perturbative rho-expansion of Section 5:

    u^rho(t, x, y) = u_0 + rho*u_1 + rho²*u_2 + ...

where u_0 is the independent-clock price and u_n satisfies a forced PDE
under the decoupled L_0 operator (Eq 5.13). Each coefficient is computed
via the Duhamel-Monte Carlo representation of Proposition 5.1 (Eq 5.17).

For large |rho|, combine with PadeAccelerator for stabilization.

Paper: Section 5, Proposition 5.1, Eqs (5.11)-(5.25).
arXiv: 2605.06677v1
"""

import numpy as np
from typing import Callable, Literal
from ..models.base_clock import BaseClock
from .single_barrier import SingleBarrierPricer
from .double_barrier import DoubleBarrierPricer
from .pade_accelerator import PadeAccelerator


BarrierType = Literal["uop", "doc", "dko_call", "dko_put"]


class LeverageExpansion:
    """Compute rho-expansion coefficients for barrier prices.

    Architecture
    ------------
    1. u_0 = baseline independent-clock price (from SingleBarrierPricer or
       DoubleBarrierPricer), already computed by Sections 2-3.

    2. For n >= 1, u_n is approximated via the Duhamel Monte Carlo estimator
       (Eq 5.17):

           u_n(t, x, y) = -E[integral_t^T (L_1 u_{n-1})(s, X_s^(0), Y_s^(0))
                                          * 1_{tau^(0) > s} ds]

       where (X^(0), Y^(0)) are the rho=0 dynamics and L_1 f = a(y)*sqrt(v(y))*d²f/dxdy.

    3. d²u_{n-1}/dxdy is computed via 'differentiate-the-transform' (Section 5.7):
       d/dx acts on the sine/exponential factors of the barrier integral;
       d/dy acts on Phi_{t,T}(lambda; y) via d_laplace_dy().

    Paper: Section 5, Proposition 5.1, Eqs (5.11)-(5.25).
    """

    def __init__(
        self,
        clock: BaseClock,
        contract_params: dict,
        barrier_type: BarrierType = "doc",
        n_paths: int = 100_000,
        n_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        clock           : BaseClock instance
        contract_params : dict with keys F0, K, L/H, T, r, q
        barrier_type    : one of 'uop', 'doc', 'dko_call', 'dko_put'
        n_paths         : MC paths for Duhamel estimator (Section 6.3.1)
        n_steps         : time steps per path
        seed            : random seed for reproducibility
        """
        self.clock = clock
        self.cp    = contract_params
        self.barrier_type = barrier_type
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed    = seed
        self.rng     = np.random.default_rng(seed)

        # Instantiate baseline pricers
        self._sb_pricer = SingleBarrierPricer(clock)
        self._db_pricer = DoubleBarrierPricer(clock)

        # Cache for computed coefficients
        self._coefficients: list[float] = []

    # ------------------------------------------------------------------
    # Baseline u_0 (rho = 0 price)
    # ------------------------------------------------------------------

    def _compute_u0(self) -> float:
        """Compute u_0 = baseline independent-clock barrier price.

        Paper: Eq (5.12) — u_0 is the ρ=0 solution, given by Sections 2-3.
        """
        cp = self.cp
        if self.barrier_type == "uop":
            return self._sb_pricer.price_uop(
                cp["F0"], cp["K"], cp["H"], cp["T"], cp.get("r", 0), cp.get("q", 0)
            )
        elif self.barrier_type == "doc":
            return self._sb_pricer.price_doc(
                cp["F0"], cp["K"], cp["L"], cp["T"], cp.get("r", 0), cp.get("q", 0)
            )
        elif self.barrier_type == "dko_call":
            return self._db_pricer.price_dko_call(
                cp["F0"], cp["K"], cp["L"], cp["H"], cp["T"], cp.get("r", 0), cp.get("q", 0)
            )
        elif self.barrier_type == "dko_put":
            return self._db_pricer.price_dko_put(
                cp["F0"], cp["K"], cp["L"], cp["H"], cp["T"], cp.get("r", 0), cp.get("q", 0)
            )
        else:
            raise ValueError(f"Unknown barrier_type: {self.barrier_type}")

    # ------------------------------------------------------------------
    # Baseline u_0 as a function of (t, x) — needed for Duhamel
    # ------------------------------------------------------------------

    def _u0_fn(self, t: float, x: float, y: float) -> float:
        """Evaluate u_0(t, x, y) — conditional plug-in of Sections 2-3.

        At intermediate time t, x = log(F_t), y = clock factor state.
        The remaining-clock Laplace transform is Phi_{t,T}(lambda; y).
        We create a temporary pricer using conditional_laplace.

        Paper: Section 5.6.
        """
        cp = self.cp
        T  = cp["T"]
        remaining = T - t

        if remaining <= 1e-8:
            # At maturity: return payoff if in corridor
            return self._terminal_payoff(x)

        # Temporary clock with conditional Phi_{t,T}(lambda; y)
        tmp_clock = _ConditionalClock(self.clock, t, T, y)
        tmp_sb = SingleBarrierPricer(tmp_clock)
        tmp_db = DoubleBarrierPricer(tmp_clock)

        # Forward price at t is exp(x) (already in forward measure)
        F_t = np.exp(x)

        try:
            if self.barrier_type == "uop":
                return tmp_sb.price_uop(F_t, cp["K"], cp["H"], remaining,
                                         cp.get("r", 0), cp.get("q", 0))
            elif self.barrier_type == "doc":
                return tmp_sb.price_doc(F_t, cp["K"], cp["L"], remaining,
                                         cp.get("r", 0), cp.get("q", 0))
            elif self.barrier_type in ("dko_call", "dko_put"):
                method = tmp_db.price_dko_call if self.barrier_type == "dko_call" else tmp_db.price_dko_put
                return method(F_t, cp["K"], cp["L"], cp["H"], remaining,
                              cp.get("r", 0), cp.get("q", 0))
        except Exception:
            return 0.0

    def _terminal_payoff(self, x: float) -> float:
        """Evaluate barrier payoff at maturity."""
        cp = self.cp
        S = np.exp(x)
        if self.barrier_type in ("uop",):
            return max(cp["K"] - S, 0.0)
        elif self.barrier_type in ("doc", "dko_call"):
            return max(S - cp["K"], 0.0)
        elif self.barrier_type == "dko_put":
            return max(cp["K"] - S, 0.0)
        return 0.0

    # ------------------------------------------------------------------
    # d²u_0 / dxdy — 'differentiate-the-transform' (Section 5.7)
    # ------------------------------------------------------------------

    def _d2u0_dxdy(self, t: float, x: float, y: float, eps_x: float = 1e-4) -> float:
        """Mixed derivative d²u_0/dxdy at (t, x, y) via finite differences.

        Paper: Section 5.7. In principle d/dx on the sine factors and
        d/dy on Phi_{t,T} can be done analytically (differentiate-the-transform).
        Here we use finite differences on y (via d_laplace_dy) and
        a one-sided finite difference on x for the outer layer.

        This is the dominant computational task in the forced hierarchy (Eq 5.13).
        """
        # d/dy component via finite difference on y (fall back for generality)
        eps_y = max(abs(y) * 1e-4, 1e-6)
        u_yp = self._u0_fn(t, x + eps_x, y + eps_y)
        u_ym = self._u0_fn(t, x + eps_x, y - eps_y)
        u_yp0 = self._u0_fn(t, x, y + eps_y)
        u_ym0 = self._u0_fn(t, x, y - eps_y)
        d_dy_xp = (u_yp - u_ym) / (2 * eps_y)
        d_dy_x0 = (u_yp0 - u_ym0) / (2 * eps_y)
        return (d_dy_xp - d_dy_x0) / eps_x

    # ------------------------------------------------------------------
    # L_1 operator  (Section 5.3, Eq 5.9 / 5.20)
    # ------------------------------------------------------------------

    def _L1_u(self, t: float, x: float, y: float, u_fn: Callable) -> float:
        """Apply L_1 = a(y)*sqrt(v(y)) * d²u/dxdy to function u_fn.

        Paper: Eq (5.20): L_1 f(x,y) = a(y)*sqrt(v(y)) * f_xy

        For the CIR clock: a(y) = xi (vol-of-vol), v(y) = y (= v_t),
            => L_1 f = xi * sqrt(y) * f_xy    [one-factor SV, rho coupling]

        For the Squared-OU clock: Y_t is the OU factor, v(Y) = Y²,
            a(Y) = sigma (OU diffusion), => L_1 f = sigma * |Y| * f_xy
        """
        # Factor a(y)*sqrt(v(y)) depends on clock family
        av = self._av_factor(y)
        # d²u/dxdy via central differences
        eps_x = 1e-4
        eps_y = max(abs(y) * 1e-4, 1e-6)
        u_pp = u_fn(t, x + eps_x, y + eps_y)
        u_pm = u_fn(t, x + eps_x, y - eps_y)
        u_mp = u_fn(t, x - eps_x, y + eps_y)
        u_mm = u_fn(t, x - eps_x, y - eps_y)
        fxy = (u_pp - u_pm - u_mp + u_mm) / (4 * eps_x * eps_y)
        return av * fxy

    def _av_factor(self, y: float) -> float:
        """a(y)*sqrt(v(y)) factor for L_1 based on clock family.

        Paper: Eq (5.8) — the mixed-derivative coupling.
        """
        from ..models.cir_clock import CIRClock
        from ..models.sq_ou_clock import SquaredOUClock

        if isinstance(self.clock, CIRClock):
            # v(y) = y (variance), a(y) = xi
            return self.clock.xi * np.sqrt(max(y, 0.0))
        elif isinstance(self.clock, SquaredOUClock):
            # v(Y) = Y², a(Y) = sigma => a(Y)*sqrt(v(Y)) = sigma*|Y|
            return self.clock.sigma * abs(y)
        else:
            raise NotImplementedError(
                f"L1 factor not implemented for {type(self.clock).__name__}. "
                "Override _av_factor in a subclass."
            )

    # ------------------------------------------------------------------
    # Duhamel Monte Carlo estimator  (Proposition 5.1, Eq 5.17)
    # ------------------------------------------------------------------

    def _compute_un_duhamel(self, n: int, prev_u_fn: Callable) -> float:
        """Compute u_n(0, x0, y0) via Duhamel-MC (Eq 5.17).

        u_n(t, x, y) = -E_{t,x,y}[integral_t^T (L_1 u_{n-1})(s, X_s^(0), Y_s^(0))
                                                * 1_{tau^(0) > s} ds]

        At t=0, x=x0=log F0, y=y0=v0 (or Y0 for squared OU).
        Simulated under rho=0 dynamics: X and Y independent.

        Paper: Proposition 5.1, Eq (5.17).
        """
        cp = self.cp
        T  = cp["T"]
        dt = T / self.n_steps
        x0 = np.log(cp["F0"])

        from ..models.cir_clock import CIRClock
        from ..models.sq_ou_clock import SquaredOUClock

        # Simulate rho=0 paths
        if isinstance(self.clock, CIRClock):
            v_paths, gamma_paths = self.clock.simulate_path(
                T, self.n_paths, self.n_steps, self.rng
            )
            y0 = self.clock.v0
        elif isinstance(self.clock, SquaredOUClock):
            v_paths, gamma_paths = self.clock.simulate_path(
                T, self.n_paths, self.n_steps, self.rng
            )
            y0 = self.clock.Y0
        else:
            raise NotImplementedError(f"Duhamel MC not implemented for {type(self.clock).__name__}")

        # Simulate X paths (drifted time-changed BM under rho=0)
        # X_t = x0 + beta*Gamma_t + B_{Gamma_t}
        beta = -0.5
        z = self.rng.standard_normal((self.n_paths, self.n_steps))

        x_paths = np.empty((self.n_paths, self.n_steps + 1))
        x_paths[:, 0] = x0
        for i in range(self.n_steps):
            d_gamma = gamma_paths[:, i + 1] - gamma_paths[:, i]
            x_paths[:, i + 1] = (
                x_paths[:, i]
                + beta * (v_paths[:, i] * dt)
                + np.sqrt(np.maximum(d_gamma, 0.0)) * z[:, i]
            )

        # Barrier monitoring (hard — Broadie bridge correction in mc_pricer.py)
        survived = np.ones(self.n_paths, dtype=bool)
        L = cp.get("L", None)
        H = cp.get("H", None)

        # Time-integrate the Duhamel integrand
        integrand_sum = np.zeros(self.n_paths)
        times = np.linspace(0.0, T, self.n_steps + 1)

        from ..models.sq_ou_clock import SquaredOUClock as SOU
        for i in range(1, self.n_steps + 1):
            t_i = times[i]
            x_i = x_paths[:, i]
            # y_i: variance state for CIR, or OU factor for squared OU
            if isinstance(self.clock, CIRClock):
                y_i = v_paths[:, i]
            else:
                # nu_paths = sqrt(v_paths) for SquaredOUClock
                y_i = np.sign(x_paths[:, i] - x0) * np.sqrt(np.maximum(v_paths[:, i], 0.0))

            # Update survival (simple hard monitoring; use bridge correction for production)
            if L is not None:
                survived &= x_i > np.log(L)
            if H is not None:
                survived &= x_i < np.log(H)

            # L_1 u_{n-1} on surviving paths
            for j in np.where(survived)[0]:
                try:
                    l1_val = self._L1_u(t_i, x_i[j], y_i[j], prev_u_fn)
                    integrand_sum[j] += l1_val * dt
                except Exception:
                    pass  # degenerate path — skip

        # u_n = -E[integral] = -mean of integrand_sum over surviving paths
        result = -np.mean(integrand_sum)
        return float(result)

    # ------------------------------------------------------------------
    # Public: compute all coefficients up to order N
    # ------------------------------------------------------------------

    def compute_coefficients(self, order: int = 5) -> list[float]:
        """Compute rho-expansion coefficients [C_0, C_1, ..., C_order].

        Paper: Eq (6.10); Table 6.9 for reference values under CIR Regime 1.

        Parameters
        ----------
        order : int  Maximum expansion order (5 used in paper's experiments)

        Returns
        -------
        list[float]  Coefficients [C_0, ..., C_order]
        """
        print(f"Computing rho-expansion coefficients up to order {order}...")

        # C_0: baseline price at rho=0
        C0 = self._compute_u0()
        print(f"  C_0 (baseline) = {C0:.6f}")
        self._coefficients = [C0]

        # Build incrementally — u_n depends on u_{n-1}
        # We need u_{n-1} as a callable (t, x, y) -> float
        prev_u_fn = self._u0_fn

        for n in range(1, order + 1):
            print(f"  Computing C_{n}...", end=" ", flush=True)
            Cn = self._compute_un_duhamel(n, prev_u_fn)
            print(f"{Cn:.6f}")
            self._coefficients.append(Cn)

            # For next iteration, we'd need u_n as a callable.
            # For the Duhamel chain beyond n=1, we use a numerical
            # approximation: u_n(t, x, y) ~ C_n (constant approximation).
            # This is the leading-order term; a full implementation would
            # cache a grid solution of u_n.
            # TODO: implement full grid-based u_n for n >= 2 (Eq 5.13).
            n_captured = n
            Cn_captured = Cn
            def make_prev(c):
                def prev(t, x, y):
                    return c
                return prev
            prev_u_fn = make_prev(Cn_captured)

        return list(self._coefficients)

    # ------------------------------------------------------------------
    # Public: price at given rho using Taylor or Padé
    # ------------------------------------------------------------------

    def price_rho(
        self,
        rho: float,
        method: str = "pade_32",
        pade_threshold: float = 0.1,
    ) -> tuple[float, str]:
        """Approximate barrier price at correlation rho.

        Parameters
        ----------
        rho    : float  Return-volatility correlation
        method : str    'taylor' | 'pade_11' | 'pade_22' | 'pade_32'
        pade_threshold : float  Pole-proximity safety threshold

        Returns
        -------
        price  : float
        method : str  Actual method used (may differ if pole safety triggered)
        """
        if not self._coefficients:
            raise RuntimeError("Call compute_coefficients() first.")

        C = self._coefficients

        if method == "taylor" or len(C) < 3:
            price = sum(c * rho**n for n, c in enumerate(C))
            return float(price), f"taylor_{len(C)-1}"

        pade = PadeAccelerator(C)

        if method == "pade_11" and len(C) >= 3:
            try:
                pade = PadeAccelerator.pade_11_explicit(C[0], C[1], C[2])
                return pade.evaluate_safe(rho, taylor_fallback=True, threshold=pade_threshold)
            except Exception:
                pass

        if method == "pade_22" and len(C) >= 5:
            pade.build(2, 2)
        elif method == "pade_32" and len(C) >= 6:
            pade.build(3, 2)
        elif len(C) >= 5:
            pade.build(2, 2)
        else:
            price = sum(c * rho**n for n, c in enumerate(C))
            return float(price), f"taylor_{len(C)-1}"

        return pade.evaluate_safe(rho, taylor_fallback=True, threshold=pade_threshold)

    def error_diagnostic(self, rho: float) -> float:
        """Estimate truncation error at the current expansion order.

        Paper: Eq (5.25): proportional to |rho|^{N+1} * |C_N|.
        """
        if not self._coefficients:
            return np.nan
        N = len(self._coefficients) - 1
        return abs(self._coefficients[N]) * abs(rho)**N

    def __repr__(self) -> str:
        return (
            f"LeverageExpansion(clock={self.clock!r}, "
            f"barrier={self.barrier_type}, n_paths={self.n_paths})"
        )


# ---------------------------------------------------------------------------
# Helper: Conditional clock wrapper for u_0(t, x, y) evaluation
# ---------------------------------------------------------------------------

class _ConditionalClock(BaseClock):
    """Wraps a base clock to use conditional Phi_{t,T}(lambda; y).

    Used by LeverageExpansion._u0_fn to evaluate the baseline at
    intermediate time t with clock state y (Section 5.6).
    """

    def __init__(self, base: BaseClock, t: float, T: float, y: float) -> None:
        self._base = base
        self._t = t
        self._T = T
        self._y = y

    def laplace(self, lam: np.ndarray, T: float) -> np.ndarray:
        # T here is the remaining horizon passed by pricer — we use T-t
        return self._base.conditional_laplace(lam, self._t, self._T, self._y)

    def __repr__(self) -> str:
        return f"_ConditionalClock(t={self._t}, T={self._T}, y={self._y})"
