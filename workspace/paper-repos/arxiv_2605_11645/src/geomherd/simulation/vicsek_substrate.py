"""
src/geomherd/simulation/vicsek_substrate.py
Vicsek self-driven particle model for out-of-domain transfer evaluation.
Paper: arXiv:2605.11645, Section 3.3.3 and Appendix G
Reference: Vicsek et al. (1995), Phys Rev Lett 75(6):1226-1229

N=600 particles, constant speed, heading alignment under angular noise eta.
Order-disorder phase transition at critical noise eta_c ≈ 1.6.
Agent graph built from k-NN (k=10) on heading sequence with binary edge weights.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


class VicsekSubstrate:
    """
    Standard Vicsek self-propelled particle model.

    Paper reference: Section 3.3.3, Appendix G
        N=600 particles, speed v0=0.3 (ASSUMED), interaction radius r=1.0 (ASSUMED).
        eta in {0.5, 1.0, 1.6, 2.0, 2.5}; eta_c ≈ 1.6.
        Herding event: V_a(t) = ||mean(v_hat_i)||  > 0.5 for 3 consecutive steps.

    Args:
        N: Number of particles (default 600)
        eta: Angular noise level
        speed: Particle speed v0 (ASSUMED: 0.3)
        radius: Interaction radius r (ASSUMED: 1.0)
        box_size: Simulation box size L (ASSUMED: L=sqrt(N) for density ≈ 1)
        seed: Random seed
    """

    def __init__(
        self,
        N: int = 600,
        eta: float = 1.6,
        speed: float = 0.3,
        radius: float = 1.0,
        box_size: Optional[float] = None,
        seed: int = 42,
    ):
        self.N = N
        self.eta = eta
        self.speed = speed
        self.radius = radius
        self.box_size = box_size or float(np.sqrt(N))  # density ≈ 1
        self._rng = np.random.default_rng(seed)
        # State
        self._positions: np.ndarray = np.zeros((N, 2))
        self._headings: np.ndarray = np.zeros(N)  # angle in radians
        self._t: int = 0
        self._snapshot_buffer = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset particles to random positions and headings."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        L = self.box_size
        self._positions = self._rng.uniform(0, L, (self.N, 2))
        self._headings = self._rng.uniform(-np.pi, np.pi, self.N)
        self._t = 0
        self._snapshot_buffer = []

    def step(self) -> Tuple[np.ndarray, Dict]:
        """
        Advance one Vicsek step.

        Update rule:
          1. Find neighbors within radius r (periodic boundary)
          2. Compute mean heading of neighbors + self
          3. Add uniform noise: theta_i <- mean_theta + U(-eta/2, eta/2)
          4. Move: x_i <- x_i + v0 * (cos(theta_i), sin(theta_i))

        Returns:
            headings: [N] float array of particle headings
            info: dict with polarisation, t
        """
        L = self.box_size
        # Periodic-boundary pairwise distances via KD-tree
        tree = cKDTree(self._positions, boxsize=L)
        # For each particle, find neighbors within radius
        new_headings = np.zeros(self.N)
        for i in range(self.N):
            neighbor_idx = tree.query_ball_point(self._positions[i], self.radius)
            # Mean heading of neighbors (including self) via circular mean
            angles = self._headings[neighbor_idx]
            mean_sin = np.sin(angles).mean()
            mean_cos = np.cos(angles).mean()
            mean_theta = np.arctan2(mean_sin, mean_cos)
            # Add angular noise
            noise = self._rng.uniform(-self.eta / 2, self.eta / 2)
            new_headings[i] = mean_theta + noise

        self._headings = new_headings
        # Move particles (periodic boundary)
        dx = self.speed * np.cos(self._headings)
        dy = self.speed * np.sin(self._headings)
        self._positions[:, 0] = (self._positions[:, 0] + dx) % L
        self._positions[:, 1] = (self._positions[:, 1] + dy) % L
        self._t += 1

        polarisation = self.get_polarisation()
        info = {"polarisation": polarisation, "t": self._t}
        return self._headings.copy(), info

    def get_polarisation(self) -> float:
        """
        V_a(t) = ||mean(v_hat_i)|| — polarisation order parameter.
        Paper: Appendix G — event tau* when Va(t) > 0.5 for 3 consecutive steps.
        """
        mean_vx = np.cos(self._headings).mean()
        mean_vy = np.sin(self._headings).mean()
        return float(np.sqrt(mean_vx**2 + mean_vy**2))

    def heading_to_actions(self, headings: np.ndarray, n_bins: int = 3) -> np.ndarray:
        """
        Discretize continuous headings into n_bins action categories.
        Used to build the agent graph (binary edge weights on heading sequences).
        ASSUMED: uniform binning of [-pi, pi] into n_bins={0,1,2} labels.
        """
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        return np.digitize(headings, bins[:-1]) - 1  # labels in {0,...,n_bins-1}

    def __repr__(self) -> str:
        return (f"VicsekSubstrate(N={self.N}, eta={self.eta}, "
                f"speed={self.speed}, radius={self.radius}, t={self._t})")
