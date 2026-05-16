"""
State sampling distribution mu_n for SPG-UVM training.

At each backward time step n, the actor and critic are trained on states
sampled from a distribution mu_n that covers the reachable state space.

From Section 4.1.3: "we sample initial conditions from a log-normal
distribution with diagonal covariance (zero correlation between assets)."

X^i_0 ~ LogNormal(log(x0) + (r - 0.5*sigma_ref^2)*t_n,  sigma_ref * sqrt(t_n))

where:
  - x0 = initial asset price (default 100)
  - sigma_ref = reference volatility (midpoint of [sigma_min, sigma_max])
  - t_n = n * dt = time elapsed from 0 to step n

Assets are sampled INDEPENDENTLY (diagonal covariance, zero correlation).
This is explicitly stated in Section 4.1.3 of arXiv:2605.06670.

Antithetic variates: for each sampled xi, also use -xi. This halves
the MC variance at essentially no extra cost (Section 4.1.3).
"""
from __future__ import annotations

import torch
from torch import Tensor


class StateSampler:
    """
    Samples asset price states X_n ~ mu_n for SPG-UVM training.

    Log-normal distribution with diagonal covariance.
    Section 4.1.3 of arXiv:2605.06670.

    Args:
        d:         Number of assets.
        x0:        Initial price (scalar, same for all assets).
        sigma_min: Per-asset minimum volatility [d].
        sigma_max: Per-asset maximum volatility [d].
        r:         Risk-free rate.
        dt:        Time step size T/N.
    """

    def __init__(
        self,
        d: int,
        x0: float,
        sigma_min: list,
        sigma_max: list,
        r: float,
        dt: float,
    ) -> None:
        self.d = d
        self.x0 = x0
        self.r = r
        self.dt = dt
        # Reference volatility: midpoint of each asset's interval
        self.sigma_ref = torch.tensor(
            [(s1 + s2) / 2.0 for s1, s2 in zip(sigma_min, sigma_max)],
            dtype=torch.float32,
        )

    def sample(
        self, n: int, n_paths: int, device: torch.device
    ) -> Tensor:
        """
        Sample n_paths states from mu_n (distribution at time step n).

        Args:
            n:        Time step index (0 = initial, N = terminal).
            n_paths:  Number of paths to sample.
            device:   Target device.

        Returns:
            X: Asset prices [n_paths, d], all > 0.
        """
        t_n = n * self.dt  # elapsed time at step n

        if t_n <= 0.0:
            # At t=0: deterministic initial price
            return torch.full((n_paths, self.d), self.x0, device=device)

        sigma_ref = self.sigma_ref.to(device)  # [d]

        # Log-normal parameters (diagonal covariance, zero correlation)
        # mu_log = log(x0) + (r - 0.5 * sigma_ref^2) * t_n
        # std_log = sigma_ref * sqrt(t_n)
        mu_log = (
            torch.log(torch.tensor(self.x0, device=device))
            + (self.r - 0.5 * sigma_ref ** 2) * t_n
        )  # [d]
        std_log = sigma_ref * (t_n ** 0.5)  # [d]

        # Sample: z ~ N(0, I), X = exp(mu_log + std_log * z)
        z = torch.randn(n_paths, self.d, device=device)
        X = torch.exp(mu_log + std_log * z)  # [n_paths, d]
        return X

    def sample_antithetic(
        self, n: int, n_paths: int, device: torch.device
    ) -> Tensor:
        """
        Sample n_paths/2 states, then mirror with antithetic variates.

        For each z sampled, we also include -z, giving n_paths total samples.
        Antithetic variates halve the variance of MC estimates (Section 4.1.3).

        Args:
            n:        Time step index.
            n_paths:  Total paths (must be even).
            device:   Target device.

        Returns:
            X: [n_paths, d] — first half original, second half antithetic.
        """
        assert n_paths % 2 == 0, "n_paths must be even for antithetic variates"
        half = n_paths // 2
        t_n = n * self.dt

        if t_n <= 0.0:
            return torch.full((n_paths, self.d), self.x0, device=device)

        sigma_ref = self.sigma_ref.to(device)
        mu_log = (
            torch.log(torch.tensor(self.x0, device=device))
            + (self.r - 0.5 * sigma_ref ** 2) * t_n
        )
        std_log = sigma_ref * (t_n ** 0.5)

        z = torch.randn(half, self.d, device=device)
        X_pos = torch.exp(mu_log + std_log * z)        # [half, d]
        X_neg = torch.exp(mu_log + std_log * (-z))     # [half, d] — antithetic
        return torch.cat([X_pos, X_neg], dim=0)        # [n_paths, d]

    def __repr__(self) -> str:
        return f"StateSampler(d={self.d}, x0={self.x0}, dt={self.dt:.4f})"
