"""
Ornstein-Uhlenbeck process simulator (Section 3, Eq. 3.3).

Uses the exact transition density (available in closed form for OU), rather
than an Euler discretization, since the exact scheme is no more expensive and
introduces no discretization bias.
"""

from __future__ import annotations

import numpy as np


class OUSimulator:
    """Simulates Ornstein-Uhlenbeck paths via exact conditional-Gaussian sampling.

    Args:
        n_steps: Number of discretization steps (paper: 100).
        T: Time horizon (paper default: 0.1).
    """

    def __init__(self, n_steps: int = 100, T: float = 0.1):
        self.n_steps = n_steps
        self.T = T
        self.dt = T / n_steps

    def simulate(self, n_paths: int, params: dict, rng: np.random.Generator) -> np.ndarray:
        """Simulate n_paths independent OU trajectories.

        Args:
            n_paths: Number of paths.
            params: Dict with keys X0, kappa, theta, sigma (scalars or arrays
                of length n_paths).
            rng: NumPy random generator.

        Returns:
            [n_paths, n_steps+1] array of OU paths.
        """
        X0 = params["X0"]
        kappa = np.broadcast_to(params["kappa"], (n_paths,)).astype(float)
        theta = np.broadcast_to(params["theta"], (n_paths,)).astype(float)
        sigma = np.broadcast_to(params["sigma"], (n_paths,)).astype(float)

        x = np.full(n_paths, X0, dtype=float)
        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = x

        exp_neg_kdt = np.exp(-kappa * self.dt)
        # exact conditional variance of OU transition: sigma^2/(2 kappa) * (1 - exp(-2 kappa dt))
        cond_var = (sigma**2) / (2 * kappa + 1e-12) * (1 - exp_neg_kdt**2)
        cond_std = np.sqrt(np.maximum(cond_var, 0.0))

        for t in range(1, self.n_steps + 1):
            mean = theta + (x - theta) * exp_neg_kdt
            z = rng.normal(0.0, 1.0, size=n_paths)
            x = mean + cond_std * z
            paths[:, t] = x

        return paths

    @staticmethod
    def sample_random_params(n_paths: int, cfg: dict, rng: np.random.Generator) -> dict:
        kappa = rng.uniform(*cfg["random"]["kappa_range"], size=n_paths)
        theta = rng.uniform(*cfg["random"]["theta_range"], size=n_paths)
        sigma = rng.uniform(*cfg["random"]["sigma_range"], size=n_paths)
        return {"X0": cfg["fixed"]["X0"], "kappa": kappa, "theta": theta, "sigma": sigma}
