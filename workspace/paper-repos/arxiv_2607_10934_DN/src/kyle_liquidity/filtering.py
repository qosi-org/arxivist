"""
Kalman-Bucy price filter: simulates the market maker's observable order flow Y_t,
equilibrium price P*_t = E[v~ | F^M_t], and posterior covariance Sigma*_t, for a GIVEN
market-depth process M*_t (Section 3, eq. (D.1)-(D.3), Proposition D.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .strategy import InsiderStrategy


@dataclass
class EquilibriumSimulator:
    """Simulates one realized equilibrium path via Euler-Maruyama discretization.

    Args:
        n_assets: dimension n of the traded-asset vector.
        M_star_fn: callable(t: float) -> np.ndarray [n, n], the market depth M*_t.
        Sigma_star_fn: callable(t: float) -> np.ndarray [n, n], the posterior covariance Sigma*_t.
        sigma_fn: callable(t: float) -> np.ndarray [n, n], the noise-volatility matrix sigma_t.
    """

    n_assets: int
    M_star_fn: Callable[[float], np.ndarray]
    Sigma_star_fn: Callable[[float], np.ndarray]
    sigma_fn: Callable[[float], np.ndarray]
    strategy: InsiderStrategy = field(default_factory=InsiderStrategy)

    def simulate(
        self,
        v_true: np.ndarray,
        p0: np.ndarray,
        T: float,
        n_steps: int,
        seed: int = 0,
        eps_boundary: float = 1e-6,
    ) -> dict:
        """
        Returns a dict with time-indexed paths:
            t: [n_steps]
            P: [n_steps, n]      -- price process P*_t
            Sigma_diag_trace: [n_steps]  -- tr(Sigma*_t), scalar summary of posterior covariance
            Y: [n_steps, n]      -- aggregate order flow Y*_t
            X: [n_steps, n]      -- insider cumulative position X*_t
            min_eig_M: [n_steps] -- min eigenvalue of M*_t along the path (empirical MDC health check)
        """
        rng = np.random.default_rng(seed)
        n = self.n_assets
        dt = T / n_steps
        # Stop strictly before T since Sigma*_T = 0 exactly (terminal revelation, eq. 3.11)
        # and the strategy's drift (eq. 3.8) divides by Sigma*_t, which is singular at T.
        t_grid = np.linspace(0.0, T - eps_boundary, n_steps)

        P = np.zeros((n_steps, n))
        Y = np.zeros((n_steps, n))
        X = np.zeros((n_steps, n))
        min_eig_M = np.zeros(n_steps)

        P[0] = p0
        for k in range(n_steps - 1):
            t = t_grid[k]
            M_t = np.atleast_2d(self.M_star_fn(t))
            Sigma_t = np.atleast_2d(self.Sigma_star_fn(t))
            sigma_t = np.atleast_2d(self.sigma_fn(t))
            min_eig_M[k] = float(np.min(np.linalg.eigvalsh(M_t)))

            drift = self.strategy.drift(v_true, P[k], M_t, Sigma_t, sigma_t)
            dB = rng.normal(size=n) * np.sqrt(dt)
            dX = drift * dt
            dZ = sigma_t @ dB
            dY = dX + dZ

            # Price update via dP_t = (M*_t)^{-1} dY_t, eq. (3.9)/(D.1)
            Lambda_t = np.linalg.inv(M_t)
            dP = Lambda_t @ dY

            X[k + 1] = X[k] + dX
            Y[k + 1] = Y[k] + dY
            P[k + 1] = P[k] + dP

        M_last = np.atleast_2d(self.M_star_fn(t_grid[-1]))
        min_eig_M[-1] = float(np.min(np.linalg.eigvalsh(M_last)))

        return {
            "t": t_grid,
            "P": P,
            "Y": Y,
            "X": X,
            "min_eig_M": min_eig_M,
        }
