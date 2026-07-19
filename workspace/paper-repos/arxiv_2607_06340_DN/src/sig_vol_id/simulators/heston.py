"""
Heston stochastic volatility model — variance process simulator (Section 3, Eq. 3.1).

Only the variance path v_t is simulated and used downstream for classification;
the price process S_t is not part of the paper's classification pipeline, so it
is omitted here (the correlation rho between the driving Brownian motions is
therefore irrelevant to any result in this repo — see SIR ambiguities[0]).

IMPLEMENTATION ASSUMPTION (SIR confidence 0.45): the paper states discretized
paths use 100 time steps but does not give the exact discretization scheme.
We use a full-truncation Euler scheme, standard practice for keeping the
square-root diffusion's variance non-negative.
"""

from __future__ import annotations

import numpy as np


class HestonSimulator:
    """Simulates Heston variance paths v_t via full-truncation Euler discretization.

    Args:
        n_steps: Number of discretization steps (paper: 100).
        T: Time horizon (paper default: 0.1).
    """

    def __init__(self, n_steps: int = 100, T: float = 0.1):
        self.n_steps = n_steps
        self.T = T
        self.dt = T / n_steps

    def simulate(self, n_paths: int, params: dict, rng: np.random.Generator) -> np.ndarray:
        """Simulate n_paths independent variance trajectories.

        Args:
            n_paths: Number of paths to simulate.
            params: Dict with keys X0, kappa, theta, nu. kappa/theta/nu may be
                scalars (fixed-parameter experiments) or arrays of length
                n_paths (random-parameter experiments, one draw per path).
            rng: NumPy random generator.

        Returns:
            [n_paths, n_steps+1] array of variance paths.
        """
        X0 = params["X0"]
        kappa = np.broadcast_to(params["kappa"], (n_paths,)).astype(float)
        theta = np.broadcast_to(params["theta"], (n_paths,)).astype(float)
        nu = np.broadcast_to(params["nu"], (n_paths,)).astype(float)

        v = np.full(n_paths, X0, dtype=float)
        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = v

        sqrt_dt = np.sqrt(self.dt)
        for t in range(1, self.n_steps + 1):
            v_pos = np.maximum(v, 0.0)  # full truncation: use max(v,0) in the diffusion coefficient
            dW = rng.normal(0.0, sqrt_dt, size=n_paths)
            v = v + kappa * (theta - v_pos) * self.dt + nu * np.sqrt(v_pos) * dW
            paths[:, t] = v

        return paths

    @staticmethod
    def sample_random_params(n_paths: int, cfg: dict, rng: np.random.Generator) -> dict:
        """Draw random (kappa, theta, nu) per path, respecting the Feller-condition
        safety margin used in the paper (Section 6.1): nu ~ U(low, min(high, 0.95*sqrt(2*kappa*theta))).
        """
        kappa = rng.uniform(*cfg["random"]["kappa_range"], size=n_paths)
        theta = rng.uniform(*cfg["random"]["theta_range"], size=n_paths)
        nu_low, nu_high = cfg["random"]["nu_range"]
        margin = cfg["random"]["feller_safety_margin"]
        nu_max = margin * np.sqrt(2 * kappa * theta)
        nu_upper = np.minimum(nu_high, nu_max)
        nu = rng.uniform(nu_low, np.maximum(nu_upper, nu_low + 1e-6))
        return {"X0": cfg["fixed"]["X0"], "kappa": kappa, "theta": theta, "nu": nu}
