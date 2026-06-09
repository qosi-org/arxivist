"""
data/signal_generator.py — Synthetic Directional Price Signal.

Implements the artificial signal described in Section 4.1 (Eq. 1 & 2):

    d_t = phi * d_{t-1} + (1-phi) * epsilon_t
    epsilon_t ~ Dirichlet(alpha(r_{t+h}))

    r_{t+h} = (p_{t+h} - p_t) / p_t,  where p_{t+h} = mean of next h mid-prices

    alpha(r_{t+h}) = (a_H, a_L, a_L)  if r < -k   (DOWN)
                   = (a_L, a_H, a_L)  if -k <= r < k (NEUTRAL)
                   = (a_L, a_L, a_H)  if r >= k   (UP)

Indices: [0] = down, [1] = neutral, [2] = up

Paper: arXiv:2301.08688 — Section 4.1, Equations 1–2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SignalGenerator:
    """Generates the synthetic directional trading signal.

    Args:
        a_H: Dirichlet concentration for the true direction class.
            Controls signal quality: 1.6=low noise, 1.3=mid, 1.1=high noise.
        a_L: Dirichlet concentration for non-true classes. Fixed at 1.0.
        phi: Exponential smoothing persistence (default 0.9 as per paper).
        horizon_h: Forward return horizon in seconds (default 10).
        threshold_k: Return magnitude threshold for classification
            (default 4e-5 as per paper).
    """

    def __init__(
        self,
        a_H: float = 1.3,
        a_L: float = 1.0,
        phi: float = 0.9,
        horizon_h: int = 10,
        threshold_k: float = 4e-5,
    ) -> None:
        self.a_H = a_H
        self.a_L = a_L
        self.phi = phi
        self.horizon_h = horizon_h
        self.threshold_k = threshold_k

        # Initialise signal as uniform distribution over 3 classes
        self._d: NDArray[np.float64] = np.array([1 / 3, 1 / 3, 1 / 3])

    def reset(self) -> None:
        """Reset signal state (call at episode start)."""
        self._d = np.array([1 / 3, 1 / 3, 1 / 3])

    def _dirichlet_params(self, r_t_h: float) -> NDArray[np.float64]:
        """Compute Dirichlet alpha vector from realised forward return.

        Implements Eq. 2 from Section 4.1.

        Args:
            r_t_h: Smoothed forward return r_{t+h}.

        Returns:
            Alpha vector [alpha_down, alpha_neutral, alpha_up].
        """
        if r_t_h < -self.threshold_k:
            # DOWN: high mass on class 0
            return np.array([self.a_H, self.a_L, self.a_L])
        elif r_t_h < self.threshold_k:
            # NEUTRAL: high mass on class 1
            return np.array([self.a_L, self.a_H, self.a_L])
        else:
            # UP: high mass on class 2
            return np.array([self.a_L, self.a_L, self.a_H])

    def step(self, mid_prices: NDArray[np.float64], t: int) -> NDArray[np.float64]:
        """Update and return the current directional signal.

        Implements Eq. 1 from Section 4.1:
            d_t = phi * d_{t-1} + (1-phi) * epsilon_t
            epsilon_t ~ Dirichlet(alpha(r_{t+h}))

        Args:
            mid_prices: Array of mid-quote prices indexed by time.
                Must have at least t + horizon_h elements.
            t: Current time index.

        Returns:
            d_t: Updated signal as 3-vector (probabilities summing to 1).
                [p_down, p_neutral, p_up]
        """
        if t + self.horizon_h >= len(mid_prices):
            # Near end of episode: return current signal unchanged
            return self._d.copy()

        # Compute smoothed forward return (Eq. 1)
        p_t = mid_prices[t]
        p_t_h = np.mean(mid_prices[t + 1 : t + self.horizon_h + 1])
        r_t_h = (p_t_h - p_t) / (p_t + 1e-10)

        # Sample epsilon from Dirichlet (Eq. 1 + 2)
        alpha = self._dirichlet_params(r_t_h)
        epsilon = np.random.dirichlet(alpha)

        # Exponential smoothing update (Eq. 1)
        self._d = self.phi * self._d + (1 - self.phi) * epsilon

        return self._d.copy()

    @property
    def current_signal(self) -> NDArray[np.float64]:
        """Return current signal without updating."""
        return self._d.copy()

    def signal_direction(self) -> int:
        """Return predicted direction as integer class.

        Returns:
            0 = down, 1 = neutral, 2 = up (argmax of d_t).
        """
        return int(np.argmax(self._d))

    def __repr__(self) -> str:
        return (
            f"SignalGenerator(a_H={self.a_H}, a_L={self.a_L}, "
            f"phi={self.phi}, h={self.horizon_h}, k={self.threshold_k})"
        )
