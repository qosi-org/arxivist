"""
Rough Bergomi variance process simulator (Section 3, Eq. 3.2).

v_t = xi * exp(eta * W^H_t - 0.5 * eta^2 * t^(2H))

IMPLEMENTATION ASSUMPTION (SIR confidence 0.6, High-severity risk in
architecture_plan.json): the paper uses an adapted GPU hybrid scheme
(Bennedsen, Lunde & Pakkanen 2017) for simulating fractional Brownian motion.
Here we instead use an exact Cholesky decomposition of the fBM covariance
matrix, Cov(W^H_s, W^H_t) = 0.5*(s^{2H} + t^{2H} - |s-t|^{2H}).

This is mathematically exact (not an approximation) and, because the paper's
discretization uses only 100 time steps, the covariance matrix is only
100x100 -- trivially cheap to factorize once and reuse (via matrix
multiplication) across arbitrarily many paths. The "slower at scale" caveat
in the architecture plan mainly matters if someone increases n_steps well
beyond 100; at the paper's actual resolution this substitute is not a
practical bottleneck.
"""

from __future__ import annotations

import numpy as np


class RoughBergomiSimulator:
    """Simulates rough Bergomi variance paths via exact Cholesky fBM.

    Args:
        n_steps: Number of discretization steps (paper: 100).
        T: Time horizon (paper default: 0.1).
    """

    def __init__(self, n_steps: int = 100, T: float = 0.1):
        self.n_steps = n_steps
        self.T = T
        self.times = np.linspace(0, T, n_steps + 1)[1:]  # exclude t=0 (W^H_0 = 0)

    def _fbm_cholesky(self, H: float) -> np.ndarray:
        """Return the Cholesky factor L (lower-triangular) of the fBM covariance
        matrix at self.times, so that W^H = L @ Z for standard normal Z.
        """
        s = self.times[:, None]
        t = self.times[None, :]
        cov = 0.5 * (s ** (2 * H) + t ** (2 * H) - np.abs(s - t) ** (2 * H))
        # add tiny jitter for numerical PD-ness
        cov += np.eye(len(self.times)) * 1e-12
        return np.linalg.cholesky(cov)

    def simulate(
        self, n_paths: int, H: float, params: dict, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate n_paths independent rough Bergomi variance trajectories.

        Args:
            n_paths: Number of paths.
            H: Hurst parameter (0 < H < 1/2 for "rough" behavior).
            params: Dict with keys xi, eta (scalars or arrays of length n_paths).
            rng: NumPy random generator.

        Returns:
            [n_paths, n_steps+1] array of variance paths (first column = xi).
        """
        xi = params["xi"]
        eta = np.broadcast_to(params["eta"], (n_paths,)).astype(float)

        L = self._fbm_cholesky(H)
        Z = rng.normal(0.0, 1.0, size=(n_paths, len(self.times)))
        W_H = Z @ L.T  # [n_paths, n_steps]

        t_pow_2H = self.times ** (2 * H)  # [n_steps]
        log_v = eta[:, None] * W_H - 0.5 * (eta[:, None] ** 2) * t_pow_2H[None, :]
        v = xi * np.exp(log_v)

        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = xi
        paths[:, 1:] = v
        return paths

    def simulate_shared_noise(
        self, n_paths: int, H_list: list[float], params: dict, rng: np.random.Generator
    ) -> dict[float, np.ndarray]:
        """Simulate rough Bergomi paths for multiple H values, reusing the SAME
        underlying standard-normal draws Z and eta samples across all H values
        (Section 6.1's experimental control: classes should differ only
        through H, not through incidental parameter/noise draws).

        Args:
            n_paths: Number of paths per H value.
            H_list: List of Hurst parameters to simulate.
            params: Dict with keys xi, eta_range (for random eta) or eta (fixed).
            rng: NumPy random generator.

        Returns:
            Dict mapping H -> [n_paths, n_steps+1] variance paths.
        """
        xi = params["xi"]
        if "eta" in params:
            eta = np.broadcast_to(params["eta"], (n_paths,)).astype(float)
        else:
            eta = rng.uniform(*params["eta_range"], size=n_paths)

        # Shared standard-normal draws: since the Cholesky factor L depends on H,
        # we share Z (not W^H itself) so that the same "random seed" per path
        # maps through each H's own covariance structure, matching the paper's
        # intent of holding eta (and the underlying noise realization) fixed
        # across H while only H varies.
        Z = rng.normal(0.0, 1.0, size=(n_paths, self.n_steps))

        results = {}
        for H in H_list:
            L = self._fbm_cholesky(H)
            W_H = Z @ L.T
            t_pow_2H = self.times ** (2 * H)
            log_v = eta[:, None] * W_H - 0.5 * (eta[:, None] ** 2) * t_pow_2H[None, :]
            v = xi * np.exp(log_v)
            paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
            paths[:, 0] = xi
            paths[:, 1:] = v
            results[H] = paths
        return results
