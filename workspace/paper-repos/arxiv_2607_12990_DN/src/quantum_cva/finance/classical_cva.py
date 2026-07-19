"""
Classical risk-neutral multi-asset CVA benchmark.

Implements Section 2.1 of arXiv:2607.12990: correlated multi-asset GBM
dynamics with piecewise-constant volatility (Eq. 2-11), Black-Scholes-Merton
pricing (Eq. 13-15), CDS-implied survival-curve bootstrap (Eq. 26), and the
continuous / finite-grid Monte Carlo CVA estimators (Eq. 28-30).

SIR reference: mathematical_spec entries for Eq. 21, 26, 28.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class Instrument:
    """One trade in the netting set (Table 3)."""

    instrument_id: int
    underlying: str
    option_type: str  # "call" | "put" | "forward"
    position: str  # "long" | "short"
    multiplier: float
    strike: float
    maturity_years: float

    @property
    def sign(self) -> float:
        return 1.0 if self.position == "long" else -1.0


class MultiAssetGBMSimulator:
    """Correlated multi-asset log-normal path simulator (Section 2.1.1).

    Implements the exact log-normal transition (Eq. 9/11), which integrates
    intra-step volatility changes exactly (no Euler discretisation bias).

    Args:
        spots: initial spot prices per underlying, keyed by name.
        dividend_yields: continuous dividend yields q_k per underlying.
        risk_free_rate: flat continuously compounded rate r.
        volatilities: piecewise-constant per-underlying volatility, keyed by
            name, as an array aligned with `monitoring_dates` (sigma_k,j from
            Eq. 2; here simplified to one value per monitoring interval,
            consistent with the paper's ATM-implied-vol bucket construction).
        correlation_matrix: d x d instantaneous correlation matrix rho_kl.
    """

    def __init__(
        self,
        spots: Dict[str, float],
        dividend_yields: Dict[str, float],
        risk_free_rate: float,
        volatilities: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
    ) -> None:
        self.underlyings = list(spots.keys())
        self.spots = spots
        self.dividend_yields = dividend_yields
        self.r = risk_free_rate
        self.volatilities = volatilities
        self.correlation_matrix = correlation_matrix
        self._cholesky = self._regularise_and_factorise(correlation_matrix)

    def __repr__(self) -> str:  # noqa: D105
        return f"MultiAssetGBMSimulator(underlyings={self.underlyings})"

    @staticmethod
    def _regularise_and_factorise(rho: np.ndarray) -> np.ndarray:
        """Cholesky-factorise rho = L L^T, applying a minimal spectral shift
        if rho fails positive-definiteness (Section 2.1.1, footnote after Eq. 11)."""
        try:
            return np.linalg.cholesky(rho)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(rho)
            eigvals_clipped = np.clip(eigvals, 1e-8, None)
            rho_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            d = np.sqrt(np.diag(rho_psd))
            rho_psd = rho_psd / np.outer(d, d)
            return np.linalg.cholesky(rho_psd)

    def simulate_paths(
        self, n_paths: int, monitoring_dates: np.ndarray, seed: int = 100000
    ) -> np.ndarray:
        """Simulate correlated log-normal paths via the exact transition (Eq. 9/11).

        Args:
            n_paths: N_MC (paper uses 2e5).
            monitoring_dates: array of monitoring dates t_1, ..., t_M.
            seed: RNG seed (paper's disclosed classical-benchmark seed is
                100000, Table 9).

        Returns:
            Array of shape [n_paths, M, d] with simulated prices at each
            monitoring date for each underlying, ordered as `self.underlyings`.
        """
        rng = np.random.default_rng(seed)
        d = len(self.underlyings)
        M = len(monitoring_dates)
        paths = np.zeros((n_paths, M, d))

        prev_prices = np.array([self.spots[u] for u in self.underlyings])
        prev_prices = np.tile(prev_prices, (n_paths, 1))
        prev_t = 0.0

        for i, t in enumerate(monitoring_dates):
            dt = t - prev_t
            sigmas = np.array(
                [self.volatilities[u][min(i, len(self.volatilities[u]) - 1)] for u in self.underlyings]
            )
            qs = np.array([self.dividend_yields[u] for u in self.underlyings])

            Z = rng.standard_normal((n_paths, d))
            correlated_Z = Z @ self._cholesky.T

            drift = (self.r - qs - 0.5 * sigmas**2) * dt
            diffusion = sigmas * np.sqrt(dt) * correlated_Z
            prev_prices = prev_prices * np.exp(drift + diffusion)
            paths[:, i, :] = prev_prices
            prev_t = t

        return paths


class BlackScholesPricer:
    """European call/put/forward pricing under constant-volatility BSM
    (Section 2.1.2, Eq. 13-15)."""

    def __repr__(self) -> str:  # noqa: D105
        return "BlackScholesPricer()"

    @staticmethod
    def _d_pm(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
        if T <= 0 or sigma <= 0:
            return np.inf, np.inf
        d_plus = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d_minus = d_plus - sigma * np.sqrt(T)
        return d_plus, d_minus

    def call_price(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """European call price c(t,T,K)  (Eq. 13)."""
        if T <= 0:
            return max(S - K, 0.0)
        d_plus, d_minus = self._d_pm(S, K, T, r, q, sigma)
        return S * np.exp(-q * T) * norm.cdf(d_plus) - K * np.exp(-r * T) * norm.cdf(d_minus)

    def put_price(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """European put price p(t,T,K)  (Eq. 14)."""
        if T <= 0:
            return max(K - S, 0.0)
        d_plus, d_minus = self._d_pm(S, K, T, r, q, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d_minus) - S * np.exp(-q * T) * norm.cdf(-d_plus)

    def forward_price(self, S: float, K: float, T: float, r: float, q: float) -> float:
        """Forward contract value f(t,T,K)  (Eq. 15)."""
        return S * np.exp(-q * T) - K * np.exp(-r * T)


class CDSBootstrapper:
    """Bootstraps a piecewise-constant hazard-rate / survival curve from CDS
    par spreads (Section 2.1.3, Eq. 26)."""

    def __repr__(self) -> str:  # noqa: D105
        return "CDSBootstrapper()"

    def bootstrap_survival_curve(
        self,
        cds_tenors: np.ndarray,
        cds_spreads: np.ndarray,
        recovery_cds: float,
        discount_fn: Callable[[float], float],
        payments_per_year: int = 4,
    ) -> Callable[[float], float]:
        """Sequentially bootstrap the survival curve P(0,u) via the par-spread
        condition (Eq. 26): PV(premium leg) = PV(protection leg).

        Args:
            cds_tenors: CDS maturities (years), increasing order.
            cds_spreads: quoted CDS par spreads (decimal, e.g. 0.0095 for 95bp).
            recovery_cds: R_CDS assumed by the CDS market (paper uses 0.40).
            discount_fn: callable t -> D(0,t).
            payments_per_year: CDS premium payment frequency (paper: quarterly).

        Returns:
            A callable u -> P(0,u), piecewise-constant-hazard survival
            probability, valid for u in [0, max(cds_tenors)].
        """
        hazard_knots = []  # (t_start, t_end, lambda)
        prev_tenor = 0.0

        def survival_so_far(u: float) -> float:
            if u <= 0:
                return 1.0
            total = 0.0
            t0 = 0.0
            for t_start, t_end, lam in hazard_knots:
                seg_end = min(u, t_end)
                if seg_end > t0:
                    total += lam * (seg_end - t0)
                t0 = t_end
                if u <= t_end:
                    break
            return np.exp(-total)

        for tenor, spread in zip(cds_tenors, cds_spreads):
            n_payments = max(1, int(round((tenor - prev_tenor) * payments_per_year)))
            payment_times = np.linspace(prev_tenor, tenor, n_payments + 1)[1:]
            dt = payment_times[1] - payment_times[0] if len(payment_times) > 1 else (tenor - prev_tenor)

            def objective(lam_candidate: float) -> float:
                trial_knots = hazard_knots + [(prev_tenor, tenor, lam_candidate)]

                def surv(u: float) -> float:
                    total = 0.0
                    t0 = 0.0
                    for t_start, t_end, lam in trial_knots:
                        seg_end = min(u, t_end)
                        if seg_end > t0:
                            total += lam * (seg_end - t0)
                        t0 = t_end
                        if u <= t_end:
                            break
                    return np.exp(-total)

                premium_leg = sum(
                    spread * dt * discount_fn(pt) * surv(pt) for pt in payment_times
                )
                protection_leg = sum(
                    (1 - recovery_cds)
                    * discount_fn(pt)
                    * (surv(payment_times[i - 1] if i > 0 else prev_tenor) - surv(pt))
                    for i, pt in enumerate(payment_times)
                )
                return premium_leg - protection_leg

            # Bisection root-find for the piecewise-constant hazard rate on this bucket
            lo, hi = 1e-6, 5.0
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if objective(lo) * objective(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            lam_fit = 0.5 * (lo + hi)
            hazard_knots.append((prev_tenor, tenor, lam_fit))
            prev_tenor = tenor

        return survival_so_far


class CVAEstimator:
    """Continuous Monte Carlo and finite-grid tabulated CVA estimators
    (Section 2.1.3, Eq. 28-30; Section 3.2.1)."""

    def __init__(self, pricer: BlackScholesPricer) -> None:
        self.pricer = pricer

    def __repr__(self) -> str:  # noqa: D105
        return "CVAEstimator()"

    def netting_set_value(
        self,
        paths_at_t: Dict[str, np.ndarray],
        instruments: List[Instrument],
        t: float,
        r: float,
        dividend_yields: Dict[str, float],
        volatilities: Dict[str, float],
    ) -> np.ndarray:
        """Mark-to-market netting-set value V(t) at a single monitoring date
        (Eq. 16), vectorised over simulated paths."""
        total = np.zeros_like(next(iter(paths_at_t.values())))
        for inst in instruments:
            S = paths_at_t[inst.underlying]
            T_remaining = max(inst.maturity_years - t, 0.0)
            q = dividend_yields[inst.underlying]
            sigma = volatilities[inst.underlying]
            if T_remaining <= 0:
                continue
            if inst.option_type == "call":
                price = np.array(
                    [self.pricer.call_price(s, inst.strike, T_remaining, r, q, sigma) for s in S]
                )
            elif inst.option_type == "put":
                price = np.array(
                    [self.pricer.put_price(s, inst.strike, T_remaining, r, q, sigma) for s in S]
                )
            else:
                price = self.pricer.forward_price(S, inst.strike, T_remaining, r, q)
            total = total + inst.sign * inst.multiplier * price
        return total

    def monte_carlo_cva(
        self,
        paths: np.ndarray,
        underlyings: List[str],
        instruments: List[Instrument],
        monitoring_dates: np.ndarray,
        survival_fn: Callable[[float], float],
        discount_fn: Callable[[float], float],
        recovery_cva: float,
        r: float,
        dividend_yields: Dict[str, float],
        volatilities: Dict[str, np.ndarray],
    ) -> Tuple[float, float]:
        """Continuous-underlying Monte Carlo CVA estimator (Eq. 30).

        Args:
            paths: simulated paths, shape [n_paths, M, d].
            underlyings: ordered list of underlying names matching paths' last axis.
            instruments: netting-set instruments (Table 3).
            monitoring_dates: t_1, ..., t_M.
            survival_fn: u -> P(0,u).
            discount_fn: t -> D(0,t).
            recovery_cva: R_CVA (paper uses 0.415).
            r: flat risk-free rate.
            dividend_yields: per-underlying q_k.
            volatilities: per-underlying sigma arrays aligned to monitoring_dates.

        Returns:
            (CVA_cont_MC estimate, standard error).
        """
        n_paths = paths.shape[0]
        losses = np.zeros(n_paths)

        prev_surv = 1.0
        for i, t in enumerate(monitoring_dates):
            paths_at_t = {u: paths[:, i, idx] for idx, u in enumerate(underlyings)}
            vols_at_t = {u: volatilities[u][min(i, len(volatilities[u]) - 1)] for u in underlyings}
            V = self.netting_set_value(paths_at_t, instruments, t, r, dividend_yields, vols_at_t)
            V_plus = np.maximum(V, 0.0)

            surv_t = survival_fn(t)
            delta_q = prev_surv - surv_t  # Eq. 27
            losses += discount_fn(t) * V_plus * delta_q
            prev_surv = surv_t

        cva_paths = (1 - recovery_cva) * losses
        cva_est = float(np.mean(cva_paths))
        cva_se = float(np.std(cva_paths) / np.sqrt(n_paths))
        return cva_est, cva_se

    def finite_grid_cva(
        self,
        prob_tensor: np.ndarray,
        exposure_tensor: np.ndarray,
        discount_vec: np.ndarray,
        default_incr_vec: np.ndarray,
        recovery_cva: float,
    ) -> float:
        """Finite-grid tabulated CVA benchmark (Eq. 31).

        CVA_Delta = (1-R_CVA) * sum_i D(0,t_i) * Delta q(t_i) * sum_j Q_D(S_ti in B_j) * V+_{i,j}

        Args:
            prob_tensor: P_{i,j} (unconditioned joint tensor, i.e. already
                including the uniform time marginal 1/M -- see grid_encoding.py);
                pass Q_D(S_ti in B_j) * M here if using the raw conditional
                probabilities from `FiniteGridBuilder.build_probability_tensor`.
            exposure_tensor: V+_{i,j}, same shape as prob_tensor.
            discount_vec: D(0,t_i), shape [M].
            default_incr_vec: Delta q(t_i), shape [M].
            recovery_cva: R_CVA.

        Returns:
            Scalar CVA_tab_Delta.
        """
        M = prob_tensor.shape[0]
        total = 0.0
        for i in range(M):
            inner = np.sum(prob_tensor[i] * exposure_tensor[i])
            total += discount_vec[i] * default_incr_vec[i] * inner
        return (1 - recovery_cva) * total
