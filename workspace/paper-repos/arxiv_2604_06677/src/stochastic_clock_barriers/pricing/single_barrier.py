"""
pricing/single_barrier.py
=========================
Semi-analytic single-barrier option pricing under an independent stochastic clock.

Implements Theorems 2.1 and 2.2 of the paper:
  - Up-and-Out Put  (UOP): Eq (2.33)
  - Down-and-Out Call (DOC): Eq (2.40)

Both reduce to a single real integral involving only elementary functions
and the Laplace transform Phi_T of the terminal clock. No complex integration
is required. The clock enters ONLY through Phi_T evaluated on the quadrature
grid, so any BaseClock-compatible model plugs in directly.

Paper: Sections 2.2–2.5, arXiv 2605.06677v1
"""

import numpy as np
from scipy.integrate import quad
from typing import Callable
from ..models.base_clock import BaseClock


# ---------------------------------------------------------------------------
# Helper: Laplace argument along the quadrature line
# ---------------------------------------------------------------------------

def _laplace_arg(u: float, beta: float = -0.5) -> float:
    """lambda = (u² + β²) / 2  where β = -1/2 (Ito drift).

    Paper: Propositions 2.1 and 2.2. Under the forward martingale
    normalization β = -1/2, so β² = 1/4.
    """
    return 0.5 * (u**2 + beta**2)


# ---------------------------------------------------------------------------
# Core: Joint Survival CDF  J_beta(x; h)  —  Proposition 2.2 / Eq (2.30)
# ---------------------------------------------------------------------------

def _joint_survival_integrand(
    u: float,
    x: float,
    h: float,
    x0: float,
    beta: float,
    phi_fn: Callable[[float], float],
) -> float:
    """Integrand of J_beta(x; h) before the leading prefactor.

    From Eq (2.30):
        integrand = sin(u*(h-x0)) * [u*cos(u*(h-x)) - beta*sin(u*(h-x))]
                    / (u² + β²)  *  Phi_T((u²+β²)/2)
    """
    lam = _laplace_arg(u, beta)
    phi_val = phi_fn(lam)
    numerator = (
        np.sin(u * (h - x0))
        * (u * np.cos(u * (h - x)) - beta * np.sin(u * (h - x)))
    )
    return numerator / (u**2 + beta**2) * phi_val


def joint_survival_cdf(
    x: float,
    h: float,
    x0: float,
    beta: float,
    phi_fn: Callable[[float], float],
    n_quad: int = 200,
    quad_limit: int = 500,
) -> float:
    """Compute J_beta(x; h) = Q^T(X_T <= x, M_T < h).

    Paper: Proposition 2.2, Eq (2.30).

    Parameters
    ----------
    x      : log-strike or evaluation point (< h)
    h      : log upper barrier
    x0     : log initial forward price
    beta   : Ito drift = -1/2
    phi_fn : callable lam -> Phi_T(lam), scalar to scalar
    n_quad : number of quadrature sub-intervals for scipy.integrate.quad

    Returns
    -------
    float : J_beta(x; h)
    """
    assert x < h, f"Require x < h, got x={x}, h={h}"

    prefactor = (2.0 / np.pi) * np.exp(beta * (x - x0))

    val, _ = quad(
        _joint_survival_integrand,
        0.0,
        np.inf,
        args=(x, h, x0, beta, phi_fn),
        limit=quad_limit,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    return prefactor * val


# ---------------------------------------------------------------------------
# Up-and-Out Put  (UOP)  —  Theorem 2.1 / Eq (2.33)
# ---------------------------------------------------------------------------

def _uop_integrand(
    u: float,
    h: float,
    k: float,
    x0: float,
    phi_fn: Callable[[float], float],
) -> float:
    """Integrand of UOP formula before leading prefactor.

    From Eq (2.33):
        sin(u*(h-x0)) * sin(u*(h-k)) / (u² + 1/4) * Phi_T((u²+1/4)/2)

    Note: β² = 1/4 for β = -1/2, simplifying the denominator.
    """
    lam = 0.5 * (u**2 + 0.25)           # (u² + β²)/2 with β=-1/2
    phi_val = phi_fn(lam)
    return (
        np.sin(u * (h - x0))
        * np.sin(u * (h - k))
        / (u**2 + 0.25)
        * phi_val
    )


class SingleBarrierPricer:
    """Price up-and-out puts and down-and-out calls under an independent clock.

    The model dependence enters only through the clock's Laplace transform
    Phi_T(lambda), making this a 'plug-in' engine for any BaseClock.

    Paper: Sections 2.4–2.5, Theorems 2.1–2.2
    """

    def __init__(self, clock: BaseClock, quad_limit: int = 500) -> None:
        """
        Parameters
        ----------
        clock      : Any BaseClock instance (CIR, SquaredOU, etc.)
        quad_limit : scipy.integrate.quad limit parameter
        """
        self.clock = clock
        self.quad_limit = quad_limit

    # ------------------------------------------------------------------
    # Internal cache: Phi_T as a scalar callable
    # ------------------------------------------------------------------

    def _make_phi_fn(self, T: float) -> Callable[[float], float]:
        """Return a scalar callable lam -> Phi_T(lam), using log-space."""
        def phi(lam: float) -> float:
            return float(self.clock.laplace(np.array([lam]), T)[0])
        return phi

    # ------------------------------------------------------------------
    # Up-and-Out Put  (Theorem 2.1, Eq 2.33)
    # ------------------------------------------------------------------

    def price_uop(
        self,
        F0: float,
        K: float,
        H: float,
        T: float,
        r: float = 0.0,
        q: float = 0.0,
    ) -> float:
        """Price an up-and-out put option.

        Contract: pays max(K - S_T, 0) at T if sup_{t<=T} S_t < H.
        Requires F0 < H (spot below upper barrier).

        Paper: Theorem 2.1, Eq (2.33):
            UOP_0 = P(0,T) * (2/pi) * sqrt(K*F0)
                    * int_0^inf sin(u*(h-x0))*sin(u*(h-k))/(u²+1/4)
                               * Phi_T((u²+1/4)/2) du

        Parameters
        ----------
        F0 : float  Initial forward price (= S0 * exp((r-q)*T) for European)
        K  : float  Strike price
        H  : float  Upper barrier (H > F0)
        T  : float  Maturity
        r  : float  Risk-free rate (used for discounting only)
        q  : float  Dividend yield
        """
        assert F0 < H, f"Require F0 < H; got F0={F0}, H={H}"
        assert K > 0 and H > 0 and F0 > 0

        h = np.log(H)
        k = np.log(K)
        x0 = np.log(F0)

        # Discount factor P(0,T) = exp(-r*T)
        discount = np.exp(-r * T)

        phi_fn = self._make_phi_fn(T)

        integral, _ = quad(
            _uop_integrand,
            0.0,
            np.inf,
            args=(h, k, x0, phi_fn),
            limit=self.quad_limit,
            epsabs=1e-10,
            epsrel=1e-10,
        )

        # Eq (2.33)
        price = discount * (2.0 / np.pi) * np.sqrt(K * F0) * integral
        return float(price)

    # ------------------------------------------------------------------
    # Down-and-Out Call  (Theorem 2.2, Eq 2.40)
    # ------------------------------------------------------------------

    def price_doc(
        self,
        F0: float,
        K: float,
        L: float,
        T: float,
        r: float = 0.0,
        q: float = 0.0,
    ) -> float:
        """Price a down-and-out call option.

        Contract: pays max(S_T - K, 0) at T if inf_{t<=T} S_t > L.
        Requires L < F0 (lower barrier below spot).

        Paper: Theorem 2.2, Eq (2.40):
            DOC_0 = P(0,T) * (2/pi) * sqrt(K*F0)
                    * int_0^inf sin(u*(x0-l))*sin(u*(k-l))/(u²+1/4)
                               * Phi_T((u²+1/4)/2) du

        Parameters
        ----------
        F0 : float  Initial forward price
        K  : float  Strike price
        L  : float  Lower barrier (L < F0)
        T  : float  Maturity
        r  : float  Risk-free rate
        q  : float  Dividend yield
        """
        assert L < F0, f"Require L < F0; got L={L}, F0={F0}"
        assert K > 0 and L > 0 and F0 > 0

        l = np.log(L)
        k = np.log(K)
        x0 = np.log(F0)

        discount = np.exp(-r * T)
        phi_fn = self._make_phi_fn(T)

        def doc_integrand(u: float) -> float:
            """Eq (2.40) integrand."""
            lam = 0.5 * (u**2 + 0.25)
            phi_val = phi_fn(lam)
            return (
                np.sin(u * (x0 - l))
                * np.sin(u * (k - l))
                / (u**2 + 0.25)
                * phi_val
            )

        integral, _ = quad(
            doc_integrand,
            0.0,
            np.inf,
            limit=self.quad_limit,
            epsabs=1e-10,
            epsrel=1e-10,
        )

        # Eq (2.40)
        price = discount * (2.0 / np.pi) * np.sqrt(K * F0) * integral
        return float(price)

    def __repr__(self) -> str:
        return f"SingleBarrierPricer(clock={self.clock!r})"
