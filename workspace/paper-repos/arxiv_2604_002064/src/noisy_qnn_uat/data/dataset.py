"""Synthetic dataset generators for the two evaluation problems used in the paper.

Implements: Architecture Plan module `data/dataset.py`.
Paper sections: Section 2.3.1 / 4.2 (Gaussian density approximation), Section 4.3
(Black-Scholes Put option pricing map).

Both datasets are fully synthetic / analytically generated -- there is no external
data download or proprietary-data dependency (see architecture_plan.json
risk_assessment, Low-severity "all datasets are synthetic").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noisy_qnn_uat.data.transforms import black_scholes_put_price, gaussian_density


@dataclass
class GaussianDensityDataset:
    """Generates the 1D Gaussian density approximation problem (Section 2.3.1, 4.2)."""

    def sample_grid(
        self, sigma: float, n_points: int = 100, x_range: tuple[float, float] = (-4.0, 4.0)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a uniform grid over x_range and evaluate the true Gaussian density.

        Args:
            sigma: standard deviation of the target Gaussian.
            n_points: number of uniformly spaced grid points (paper default: 100).
            x_range: (min, max) domain, paper default (-4, 4).

        Returns:
            (x, y): x of shape [n_points], y = f_sigma(x) of shape [n_points].
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = gaussian_density(x, sigma)
        return x, y


@dataclass
class BlackScholesPutDataset:
    """Generates the 5D Black-Scholes Put pricing map (Section 4.3)."""

    def sample_training_grid(self, ranges: dict) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random grid of (S,K,T,r,sigma) within the given ranges and price them.

        Args:
            ranges: dict with keys 'S_range','K_range','T_range','r_range','sigma_range',
                each a (min, max) tuple, and optionally 'n_samples' (default 1600 to
                match the paper's 40x40=1600-point evaluation grid).

        Returns:
            (x, y): x of shape [n_samples, 5] (raw, un-normalised S,K,T,r,sigma),
                y of shape [n_samples] (true Black-Scholes Put prices).
        """
        n_samples = ranges.get("n_samples", 1600)
        rng = np.random.default_rng(ranges.get("seed", 0))

        s = rng.uniform(*ranges["S_range"], size=n_samples)
        k = rng.uniform(*ranges["K_range"], size=n_samples)
        t = rng.uniform(*ranges["T_range"], size=n_samples)
        r = rng.uniform(*ranges["r_range"], size=n_samples)
        sigma = rng.uniform(*ranges["sigma_range"], size=n_samples)

        x = np.stack([s, k, t, r, sigma], axis=1)
        y = black_scholes_put_price(s, k, t, r, sigma)
        return x, y

    def sample_eval_grid(
        self,
        k_range: tuple[float, float],
        sigma_sqrt_t_range: tuple[float, float],
        grid_size: int = 40,
        s0: float = 100.0,
        r: float = 0.03,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the 40x40 evaluation grid over (K, sigma*sqrt(T)) at fixed S0, r
        (Section 4.3: "evaluated on a 40x40 grid over (K, sigma*sqrt(T)) in
        [85,115]x[0.05,0.35] at fixed S0=100, r=0.03").

        For each (K, sigma*sqrt(T)) grid point, T is fixed at 1.0 (year) and sigma is
        derived as sigma = (sigma*sqrt(T)) / sqrt(T); this is one convention consistent
        with the paper's (K, sigma*sqrt(T)) axis choice, not the only possible one --
        see architecture_plan.json for related caveats on unspecified evaluation
        details.

        Args:
            k_range: (K_min, K_max), paper default (85, 115).
            sigma_sqrt_t_range: (min, max) of sigma*sqrt(T), paper default (0.05, 0.35).
            grid_size: grid resolution per axis (paper default 40).
            s0: fixed spot price (paper default 100).
            r: fixed risk-free rate (paper default 0.03).

        Returns:
            (x, y): x of shape [grid_size*grid_size, 5] (S0,K,T,r,sigma columns),
                y of shape [grid_size*grid_size] (true Black-Scholes Put prices).
        """
        k_vals = np.linspace(k_range[0], k_range[1], grid_size)
        sigma_sqrt_t_vals = np.linspace(sigma_sqrt_t_range[0], sigma_sqrt_t_range[1], grid_size)
        kk, ss = np.meshgrid(k_vals, sigma_sqrt_t_vals, indexing="ij")

        t_fixed = 1.0
        sigma_grid = ss / np.sqrt(t_fixed)
        k_flat = kk.ravel()
        sigma_flat = sigma_grid.ravel()
        n = k_flat.shape[0]

        s_col = np.full(n, s0)
        t_col = np.full(n, t_fixed)
        r_col = np.full(n, r)

        x = np.stack([s_col, k_flat, t_col, r_col, sigma_flat], axis=1)
        y = black_scholes_put_price(s_col, k_flat, t_col, r_col, sigma_flat)
        return x, y
