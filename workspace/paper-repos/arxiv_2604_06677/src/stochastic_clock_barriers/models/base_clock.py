"""
models/base_clock.py
====================
Abstract base class for stochastic clock families.

Every concrete clock must expose a real-axis Laplace transform
    Phi_T(lambda) = E[exp(-lambda * Gamma_T)]
and an optional conditional variant for the leverage layer (Section 5).

Paper: Section 4, arXiv 2605.06677v1
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseClock(ABC):
    """Abstract stochastic clock: Gamma_T = integral_0^T v_s ds.

    Subclasses implement clock families from Section 4 of the paper.
    All pricing formulas depend on the clock only through Phi_T evaluated
    on a real grid — this is the 'plug-in' interface.
    """

    @abstractmethod
    def laplace(self, lam: np.ndarray, T: float) -> np.ndarray:
        """Compute Phi_T(lambda) = E[exp(-lambda * Gamma_T)].

        Paper: Eq (1.2) / (2.10).

        Parameters
        ----------
        lam : np.ndarray
            Non-negative Laplace argument(s). Shape: (N,) or scalar.
        T : float
            Maturity / horizon.

        Returns
        -------
        np.ndarray
            Phi_T(lam), same shape as lam. Values in (0, 1].
        """

    def log_laplace(self, lam: np.ndarray, T: float) -> np.ndarray:
        """Return log Phi_T(lambda) for numerical stability.

        Paper: Section 6.2 — 'Phi_T is computed in log form and cached'.
        """
        return np.log(self.laplace(lam, T))

    def conditional_laplace(
        self, lam: np.ndarray, t: float, T: float, y: float
    ) -> np.ndarray:
        """Phi_{t,T}(lambda; y) = E[exp(-lambda * Gamma_{t,T}) | Y_t = y].

        Paper: Eq (5.19). Required for leverage expansion (Section 5).
        Default: falls back to unconditional (valid only for time-homogeneous
        clocks with deterministic initial state matching y).

        Subclasses should override for proper conditional evaluation.
        """
        # Default: remaining horizon only (approximate; subclasses override)
        return self.laplace(lam, T - t)

    def d_laplace_dy(self, lam: np.ndarray, T: float) -> np.ndarray:
        """Derivative of Phi_T(lambda; y) w.r.t. clock state y at t=0.

        Paper: Eq (6.8) for CIR case. Used to construct forcing terms
        in the leverage expansion (Section 5.7).

        Returns
        -------
        np.ndarray
            d/dy Phi_T(lam), same shape as lam.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented d_laplace_dy. "
            "Required for leverage expansion (Section 5)."
        )

    def simulate_gamma(
        self, T: float, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate terminal clock Gamma_T paths for Monte Carlo validation.

        Parameters
        ----------
        T : float
        n_paths : int
        n_steps : int
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Simulated Gamma_T values.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented simulate_gamma."
        )

    def simulate_path(
        self, T: float, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate (variance path, clock path) for full MC pricing.

        Returns
        -------
        v_paths : np.ndarray, shape (n_paths, n_steps+1)
        gamma_paths : np.ndarray, shape (n_paths, n_steps+1)
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented simulate_path."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
