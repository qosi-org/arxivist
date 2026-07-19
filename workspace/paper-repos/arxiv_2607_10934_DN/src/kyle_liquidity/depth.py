"""
Closed-form martingale depth (M*_t) and posterior covariance (Sigma*_t) solvers.

Implements Section 5 of arXiv:2607.10934 for the cases where the martingale dual
condition (MDC, Definition 3.1) is verified in closed form:

  - KyleConstantVolDepth       -> Section 5.1, recovers Kyle (1985)
  - BackPedersenDepth          -> Section 5.2, recovers Back & Pedersen (1998), static info
  - CollinDufresneFosDepth     -> Section 5.4, recovers Collin-Dufresne & Fos (2016)
  - CommonEigenbasisDepth      -> Section 5.6 / 5.3, recovers Back-Cocquemas-Ekren-Lioui (2020)

All quantities are as defined in the SIR's mathematical_spec (see sir-registry).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import sqrtm


@dataclass
class KyleConstantVolDepth:
    """Scalar Kyle (1985) benchmark: constant sigma, prior variance Sigma_0, horizon T.

    Eq. (5.3)-(5.6) of the paper.
    """

    Sigma_0: float
    sigma: float
    T: float = 1.0

    def M_star(self, t: np.ndarray) -> np.ndarray:
        # Eq. (5.4): M*_t = sigma / sqrt(Sigma_0), constant in t.
        return np.full_like(np.asarray(t, dtype=float), self.sigma / np.sqrt(self.Sigma_0))

    def Sigma_star(self, t: np.ndarray) -> np.ndarray:
        # Eq. (5.3): Sigma*(t) = Sigma_0 * (1 - t/T).
        t = np.asarray(t, dtype=float)
        return self.Sigma_0 * (1.0 - t / self.T)

    def lambda_star(self, t: np.ndarray) -> np.ndarray:
        # Eq. (5.5): price-impact lambda*_t = sqrt(Sigma_0) / sigma = 1 / M*_t.
        return 1.0 / self.M_star(t)


@dataclass
class BackPedersenDepth:
    """Back & Pedersen (1998), static-information case: deterministic sigma(t).

    Eq. (5.9)-(5.13) of the paper.
    """

    Sigma_0: float
    sigma_fn: "callable"  # sigma(t) -> float, deterministic
    T: float
    n_grid: int = 2000

    def _sigma_sq_integral(self, upper: float) -> float:
        grid = np.linspace(0.0, upper, max(self.n_grid, 2))
        vals = np.array([self.sigma_fn(s) ** 2 for s in grid])
        return float(np.trapezoid(vals, grid))

    def M_star(self, t: np.ndarray) -> np.ndarray:
        # Eq. (5.11): M* = sqrt( int_0^T sigma(s)^2 ds / Sigma_0 ), constant in t.
        total = self._sigma_sq_integral(self.T)
        val = np.sqrt(total / self.Sigma_0)
        return np.full_like(np.asarray(t, dtype=float), val)

    def Sigma_star(self, t: np.ndarray) -> np.ndarray:
        # Eq. (5.9): Sigma*(t) = Sigma_0 * int_t^T sigma(s)^2 ds / int_0^T sigma(s)^2 ds.
        total = self._sigma_sq_integral(self.T)
        out = np.empty_like(np.asarray(t, dtype=float))
        for i, ti in enumerate(np.atleast_1d(t)):
            remaining = total - self._sigma_sq_integral(float(ti))
            out.flat[i] = self.Sigma_0 * remaining / total
        return out


@dataclass
class CollinDufresneFosDepth:
    """Collin-Dufresne & Fos (2016), deterministic-growth stochastic liquidity.

    sigma_t follows d(sigma_t) = sigma_t m(t) dt + sigma_t nu(t, sigma_t) dW_t, m(t) deterministic.
    Implements the closed form eq. (5.22)/(5.27), which depends on the REALIZED sigma path
    (the formula is exact path-by-path, not just in expectation) -- see Section 5.4.2.
    """

    Sigma_0: float
    m_fn: "callable"  # m(t) -> float, deterministic drift of log(sigma_t)
    T: float
    n_grid: int = 2000

    def _int_m(self, upper: float) -> float:
        grid = np.linspace(0.0, upper, max(self.n_grid, 2))
        vals = np.array([self.m_fn(s) for s in grid])
        return float(np.trapezoid(vals, grid))

    def _normalizing_constant(self) -> float:
        # sqrt( (1/Sigma_0) * int_0^T exp(2 int_0^t m(s) ds) dt ), from eq. (5.22).
        grid = np.linspace(0.0, self.T, max(self.n_grid, 2))
        integrand = np.array([np.exp(2.0 * self._int_m(t)) for t in grid])
        integral = float(np.trapezoid(integrand, grid))
        return np.sqrt(integral / self.Sigma_0)

    def M_star(self, t: np.ndarray, sigma_path_fn: "callable") -> np.ndarray:
        """
        Eq. (5.22): M*_t = sigma_t * exp(-int_0^t m(s) ds) * K,
        where K is the constant computed in `_normalizing_constant`.

        `sigma_path_fn(t)` must return the REALIZED sigma_t along the simulated path
        (not its unconditional law), since eq. (5.22) is a path-by-path identity.
        """
        K = self._normalizing_constant()
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(t)
        for i, ti in enumerate(t):
            out[i] = sigma_path_fn(ti) * np.exp(-self._int_m(float(ti))) * K
        return out


@dataclass
class CommonEigenbasisDepth:
    """Multi-asset common-eigenbasis case (Section 5.6), recovers Back-Cocquemas-Ekren-Lioui (2020)
    in the constant-volatility specialization (Section 5.3).

    C and sigma_t are assumed simultaneously diagonalizable: C = V diag(Sigma0_i) V^T,
    sigma_t = V diag(sigma_i(t)) V^T for a fixed orthogonal V. The dual problem then
    decouples into n independent scalar problems (Proposition 5.2, eq. 5.49).
    """

    V: np.ndarray  # [n, n] orthogonal eigenbasis
    scalar_depths: list  # length-n list of scalar depth objects (e.g. KyleConstantVolDepth)

    def M_star(self, t: np.ndarray) -> np.ndarray:
        """Returns M*_t as an [len(t), n, n] array (or [n, n] if t is scalar)."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        n = self.V.shape[0]
        out = np.zeros((len(t_arr), n, n))
        for i, ti in enumerate(t_arr):
            diag_vals = np.array([sd.M_star(np.array([ti]))[0] for sd in self.scalar_depths])
            out[i] = self.V @ np.diag(diag_vals) @ self.V.T
        return out if np.ndim(t) else out[0]

    def Sigma_star(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        n = self.V.shape[0]
        out = np.zeros((len(t_arr), n, n))
        for i, ti in enumerate(t_arr):
            diag_vals = np.array([sd.Sigma_star(np.array([ti]))[0] for sd in self.scalar_depths])
            out[i] = self.V @ np.diag(diag_vals) @ self.V.T
        return out if np.ndim(t) else out[0]


def constant_matrix_depth(sigma: np.ndarray, C: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Section 5.3 (BCEL 2020) closed form, eq. right after Proposition 5.1:

        M* = sigma @ sqrtm(sigma @ C @ sigma)^{-1} @ sigma

    A direct matrix computation (does not require a common eigenbasis); provided for
    cross-checking `CommonEigenbasisDepth` when sigma happens to be constant.
    """
    inner = sqrtm(sigma @ C @ sigma)
    inner = np.real_if_close(inner)
    return sigma @ np.linalg.inv(inner) @ sigma
