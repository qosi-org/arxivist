"""
pricing/double_barrier.py
=========================
Semi-analytic double knock-out (DKO) option pricing under an independent clock.

Implements Theorem 3.1 of the paper:
  DKO call priced as a real Dirichlet sine series (Eq 3.9).
  Model dependence enters ONLY through Phi_T on the discrete Laplace grid.

The series is rapidly convergent (O(1/n²) decay of terms) and requires
only elementary operations plus Phi_T at the fixed grid {lambda_n}_{n>=1}.

Paper: Section 3, Theorem 3.1, Eqs (3.2)-(3.9).
arXiv: 2605.06677v1
"""

import numpy as np
from ..models.base_clock import BaseClock


# ---------------------------------------------------------------------------
# Payoff projection coefficients  A_n  (Eqs 3.5-3.8)
# ---------------------------------------------------------------------------

def _F_alpha_n(alpha: float, omega_n: float, l: float, x: float) -> float:
    """Antiderivative F_{alpha,n}(x) = int e^{alpha*x} sin(omega_n*(x-l)) dx.

    Paper: Eq (3.6):
        F_{alpha,n}(x) = e^{alpha*x} / (alpha² + omega_n²)
                         * (alpha*sin(omega_n*(x-l)) - omega_n*cos(omega_n*(x-l)))
    """
    denom = alpha**2 + omega_n**2
    phase = omega_n * (x - l)
    return (np.exp(alpha * x) / denom) * (
        alpha * np.sin(phase) - omega_n * np.cos(phase)
    )


def compute_An(
    n: int,
    l: float,
    h: float,
    k: float,
    beta: float = -0.5,
) -> float:
    """Compute the n-th payoff projection coefficient A_n.

    Paper: Eqs (3.5)-(3.8). For a call payoff (e^x - K)^+ on [l, h]:

        A_n = [F_{beta+1, n}(x)]_{c}^{h} - K * [F_{beta, n}(x)]_{c}^{h}
        c = max(k, l)

    Since omega_n*(h-l) = n*pi:
        sin(omega_n*(h-l)) = 0
        cos(omega_n*(h-l)) = (-1)^n

    So Eq (3.8): F_{alpha,n}(h) = (-1)^{n+1} * omega_n * e^{alpha*h} / (alpha² + omega_n²)

    Parameters
    ----------
    n     : int    Term index (n >= 1)
    l     : float  Log lower barrier
    h     : float  Log upper barrier
    k     : float  Log strike
    beta  : float  Ito drift = -1/2
    """
    omega_n = n * np.pi / (h - l)      # Dirichlet eigenfrequency (Eq 3.2)
    c = max(k, l)                        # lower integration limit (call payoff)

    if c >= h:
        # Strike above upper barrier: option is worthless (Eq 3.5 convention)
        return 0.0

    # F_{beta+1, n}(h) via Eq (3.8): (-1)^{n+1} * omega_n * e^{(beta+1)*h} / ((beta+1)^2 + omega_n^2)
    F_b1_h = (-1)**(n + 1) * omega_n * np.exp((beta + 1) * h) / ((beta + 1)**2 + omega_n**2)
    F_b1_c = _F_alpha_n(beta + 1, omega_n, l, c)

    F_b_h  = (-1)**(n + 1) * omega_n * np.exp(beta * h) / (beta**2 + omega_n**2)
    F_b_c  = _F_alpha_n(beta, omega_n, l, c)

    # Eq (3.7)
    An = (F_b1_h - F_b1_c) - np.exp(k) * (F_b_h - F_b_c)
    return float(An)


def compute_An_put(
    n: int,
    l: float,
    h: float,
    k: float,
    beta: float = -0.5,
) -> float:
    """Payoff projection for DKO put payoff (K - e^x)^+.

    Paper: Section 3.4 (iii), Eq (3.20):
        A_n^put = int_l^d (K - e^x) e^{beta*x} sin(omega_n*(x-l)) dx
        d = min(k, h)
    """
    omega_n = n * np.pi / (h - l)
    d = min(k, h)

    if d <= l:
        return 0.0

    # For put: K * F_{beta, n}(x) - F_{beta+1, n}(x) evaluated at [l, d]
    F_b_d = _F_alpha_n(beta, omega_n, l, d)
    F_b_l = _F_alpha_n(beta, omega_n, l, l)    # = 0 since sin(0) = 0

    F_b1_d = _F_alpha_n(beta + 1, omega_n, l, d)
    F_b1_l = _F_alpha_n(beta + 1, omega_n, l, l)  # = 0

    An_put = np.exp(k) * (F_b_d - F_b_l) - (F_b1_d - F_b1_l)
    return float(An_put)


# ---------------------------------------------------------------------------
# Main DKO Pricer
# ---------------------------------------------------------------------------

class DoubleBarrierPricer:
    """Price double knock-out (DKO) options under an independent stochastic clock.

    Uses the real Dirichlet sine series of Theorem 3.1 (Eq 3.9):

        DKO_0 = P(0,T) * (2/a) * exp(-beta*x0)
                * sum_{n=1}^{N} sin(omega_n*(x0-l)) * A_n * Phi_T(lambda_n)

    where:
        omega_n = n*pi/a    (Dirichlet eigenfrequencies, Eq 3.2)
        lambda_n = (omega_n² + beta²)/2  (Laplace grid, Eq 3.3)
        A_n = payoff projection coefficients (Eqs 3.5-3.8)

    The grid {lambda_n} is fixed for given (l, h, T). Hence {Phi_T(lambda_n)}
    can be precomputed and cached across strikes and calibration iterations.

    Paper: Section 3, Theorem 3.1
    """

    def __init__(self, clock: BaseClock, n_terms: int = 100) -> None:
        """
        Parameters
        ----------
        clock   : BaseClock instance
        n_terms : Number of sine series terms (default 100; convergence is
                  O(1/n²) so 50–100 is typically more than sufficient)
        """
        self.clock = clock
        self.n_terms = n_terms
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Grid computation
    # ------------------------------------------------------------------

    def _compute_grids(
        self, l: float, h: float, beta: float = -0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Dirichlet eigenfrequencies and Laplace grid.

        Returns
        -------
        omega : np.ndarray, shape (N,)  — eigenfrequencies (Eq 3.2)
        lam   : np.ndarray, shape (N,)  — Laplace grid (Eq 3.3 / 3.4)
        """
        a = h - l
        n = np.arange(1, self.n_terms + 1, dtype=float)
        omega = n * np.pi / a                              # Eq (3.2)
        lam   = 0.5 * (omega**2 + beta**2)                # Eq (3.3)
        return omega, lam

    def _get_cached_phi(
        self, l: float, h: float, T: float, beta: float = -0.5
    ) -> np.ndarray:
        """Return cached Phi_T(lambda_n) for the given corridor and maturity.

        Paper: Section 3.4 (i) — '{Phi_T(lambda_n)} can be precomputed once
        and reused across strikes K, payoffs, and calibration iterations.'
        """
        key = (l, h, T, beta, self.n_terms)
        if key not in self._cache:
            _, lam = self._compute_grids(l, h, beta)
            self._cache[key] = self.clock.laplace(lam, T)
        return self._cache[key]

    def clear_cache(self) -> None:
        """Clear the Phi_T cache (call when clock parameters change)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # DKO Call  (Theorem 3.1, Eq 3.9)
    # ------------------------------------------------------------------

    def price_dko_call(
        self,
        F0: float,
        K: float,
        L: float,
        H: float,
        T: float,
        r: float = 0.0,
        q: float = 0.0,
        beta: float = -0.5,
    ) -> float:
        """Price a double knock-out call option.

        Contract: pays max(S_T - K, 0) at T if L < inf S_t and sup S_t < H.

        Paper: Theorem 3.1, Eq (3.9):
            DKO_0 = P(0,T) * (2/a) * exp(-beta*x0)
                    * sum_{n=1}^N sin(omega_n*(x0-l)) * A_n * Phi_T(lambda_n)

        Parameters
        ----------
        F0 : float  Initial forward price (L < F0 < H required)
        K  : float  Strike
        L  : float  Lower barrier
        H  : float  Upper barrier
        T  : float  Maturity
        r  : float  Risk-free rate
        q  : float  Dividend yield
        beta : float  Ito drift = -1/2
        """
        assert L < F0 < H, f"Require L < F0 < H; got L={L}, F0={F0}, H={H}"

        l   = np.log(L)
        h   = np.log(H)
        k   = np.log(K)
        x0  = np.log(F0)
        a   = h - l
        discount = np.exp(-r * T)

        omega, _ = self._compute_grids(l, h, beta)
        phi_vals = self._get_cached_phi(l, h, T, beta)   # shape (N,)

        # Payoff coefficients A_n (Eqs 3.5-3.8) — independent of clock
        A_vals = np.array([
            compute_An(n + 1, l, h, k, beta) for n in range(self.n_terms)
        ])

        # Sine factors sin(omega_n*(x0 - l))
        sin_vals = np.sin(omega * (x0 - l))

        # Series summation (Eq 3.9)
        series = np.sum(sin_vals * A_vals * phi_vals)

        price = discount * (2.0 / a) * np.exp(-beta * x0) * series
        return float(price)

    # ------------------------------------------------------------------
    # DKO Put  (Section 3.4 (iii), Eq 3.20)
    # ------------------------------------------------------------------

    def price_dko_put(
        self,
        F0: float,
        K: float,
        L: float,
        H: float,
        T: float,
        r: float = 0.0,
        q: float = 0.0,
        beta: float = -0.5,
    ) -> float:
        """Price a double knock-out put option.

        Paper: Section 3.4 (iii), Eq (3.20).
        Same series as DKO call, but with put payoff coefficients A_n^put.
        """
        assert L < F0 < H

        l  = np.log(L)
        h  = np.log(H)
        k  = np.log(K)
        x0 = np.log(F0)
        a  = h - l
        discount = np.exp(-r * T)

        omega, _ = self._compute_grids(l, h, beta)
        phi_vals = self._get_cached_phi(l, h, T, beta)

        A_put_vals = np.array([
            compute_An_put(n + 1, l, h, k, beta) for n in range(self.n_terms)
        ])

        sin_vals = np.sin(omega * (x0 - l))
        series = np.sum(sin_vals * A_put_vals * phi_vals)

        price = discount * (2.0 / a) * np.exp(-beta * x0) * series
        return float(price)

    # ------------------------------------------------------------------
    # Convergence diagnostic
    # ------------------------------------------------------------------

    def convergence_check(
        self,
        F0: float, K: float, L: float, H: float, T: float,
        r: float = 0.0, check_terms: list[int] = None,
    ) -> dict:
        """Compute DKO price at several truncation points to diagnose convergence.

        Returns dict mapping n_terms -> price for each entry in check_terms.
        """
        if check_terms is None:
            check_terms = [10, 25, 50, 100, 200]

        orig_n = self.n_terms
        results = {}
        for nt in check_terms:
            self.n_terms = nt
            self.clear_cache()
            results[nt] = self.price_dko_call(F0, K, L, H, T, r)
        self.n_terms = orig_n
        self.clear_cache()
        return results

    def __repr__(self) -> str:
        return f"DoubleBarrierPricer(clock={self.clock!r}, n_terms={self.n_terms})"
