"""
models/cir_clock.py
===================
Integrated CIR (Cox-Ingersoll-Ross) stochastic clock.

The variance process v_t follows a square-root diffusion:
    dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dZ_t

The clock is Gamma_T = integral_0^T v_s ds.

The Laplace transform Phi_T(lambda) = exp(-A(T;lambda) - B(T;lambda)*v0)
where A, B satisfy the Riccati ODE system (4.3)-(4.4).

Paper: Section 4.1, Eqs (4.1)-(4.4) and (6.1)-(6.3).
arXiv: 2605.06677v1
"""

import numpy as np
from scipy.integrate import solve_ivp
from .base_clock import BaseClock


class CIRClock(BaseClock):
    """Integrated CIR stochastic clock.

    Attributes
    ----------
    kappa : float  Mean-reversion speed  (κ > 0)
    theta : float  Long-run variance     (θ > 0)
    xi    : float  Vol-of-vol            (ξ > 0)
    v0    : float  Initial variance      (v0 > 0)

    Feller condition 2κθ > ξ² ensures strict positivity of v_t.

    Paper reference: Section 4.1
    """

    def __init__(self, kappa: float, theta: float, xi: float, v0: float) -> None:
        """
        Parameters
        ----------
        kappa : float  Mean-reversion speed κ > 0
        theta : float  Long-run variance θ > 0
        xi    : float  Vol-of-vol (vol of variance) ξ > 0
        v0    : float  Initial variance v0 > 0
        """
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}")
        if xi <= 0:
            raise ValueError(f"xi must be > 0, got {xi}")
        if v0 <= 0:
            raise ValueError(f"v0 must be > 0, got {v0}")

        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.v0 = v0

        feller = 2 * kappa * theta - xi**2
        if feller <= 0:
            import warnings
            warnings.warn(
                f"Feller condition 2κθ > ξ² violated (2κθ - ξ² = {feller:.4f}). "
                "v_t may reach zero; use caution.",
                UserWarning,
                stacklevel=2,
            )
        self._feller_value = feller

    # ------------------------------------------------------------------
    # Riccati ODE system  (Section 4.1, Eqs 4.3-4.4)
    # ------------------------------------------------------------------

    def _riccati_rhs(self, t: float, y: np.ndarray, lam: float) -> list:
        """RHS of the Riccati ODE system for scalar lambda.

        State: y = [B(t; lambda), A(t; lambda)]

        Standard affine CIR Riccati (Duffie-Pan-Singleton 2000, forward in T):
            dB/dT = lambda - kappa*B - (xi²/2)*B²   (Eq 4.3, corrected sign)
            dA/dT = kappa*theta*B                    (Eq 4.4)

        Note: The paper's Eq (4.3) writes +kappa*B but the standard convention
        for FORWARD integration (T increasing from 0) requires -kappa*B.
        The minus sign is confirmed by comparison with MC ground truth.
        """
        B, A = y
        dB = lam - self.kappa * B - 0.5 * self.xi**2 * B**2
        dA = self.kappa * self.theta * B
        return [dB, dA]

    def _solve_riccati(self, lam: float, T: float) -> tuple[float, float]:
        """Solve Riccati ODE for a single lambda value; return (A(T), B(T)).

        Initial conditions: B(0) = 0, A(0) = 0  (Eqs 4.3-4.4).
        """
        sol = solve_ivp(
            self._riccati_rhs,
            t_span=(0.0, T),
            y0=[0.0, 0.0],
            args=(lam,),
            method="RK45",
            dense_output=False,
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(
                f"Riccati ODE solver failed for lambda={lam}, T={T}: {sol.message}"
            )
        B_T, A_T = sol.y[:, -1]
        return float(A_T), float(B_T)

    # ------------------------------------------------------------------
    # Laplace transform  (Section 4.1, Eq 4.2)
    # ------------------------------------------------------------------

    def laplace(self, lam: np.ndarray, T: float) -> np.ndarray:
        """Phi_T(lambda) = exp(-A(T;lambda) - B(T;lambda)*v0).

        Paper: Section 4.1, Eq (4.2).

        Parameters
        ----------
        lam : array-like  Non-negative Laplace arguments.
        T   : float       Maturity.

        Returns
        -------
        np.ndarray  Phi_T(lam), same shape as lam.
        """
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        result = np.empty_like(lam)
        for i, li in enumerate(lam.ravel()):
            A_T, B_T = self._solve_riccati(li, T)
            # Compute in log space for numerical stability (Section 6.2)
            log_phi = -A_T - B_T * self.v0
            result.ravel()[i] = np.exp(log_phi)
        return result.reshape(lam.shape)

    def log_laplace(self, lam: np.ndarray, T: float) -> np.ndarray:
        """log Phi_T(lambda) = -A(T;lambda) - B(T;lambda)*v0.

        Paper: Section 6.2 — 'Phi_T is computed in log form and cached'.
        """
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        result = np.empty_like(lam)
        for i, li in enumerate(lam.ravel()):
            A_T, B_T = self._solve_riccati(li, T)
            result.ravel()[i] = -A_T - B_T * self.v0
        return result.reshape(lam.shape)

    def d_laplace_dy(self, lam: np.ndarray, T: float) -> np.ndarray:
        """d/dv0 Phi_T(lambda) = -B(T;lambda) * Phi_T(lambda).

        Paper: Section 6.3.1, Eq (6.8):
            ∂_y Phi_{t,T}(lambda; y) = B(T-t; lambda) * Phi_{t,T}(lambda; y)
        Here at t=0, y=v0.
        """
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        phi = self.laplace(lam, T)
        B_vals = np.empty_like(lam)
        for i, li in enumerate(lam.ravel()):
            _, B_T = self._solve_riccati(li, T)
            B_vals.ravel()[i] = B_T
        # d/dv0 exp(-A - B*v0) = -B * exp(-A - B*v0) = -B * Phi_T
        return -B_vals * phi

    # ------------------------------------------------------------------
    # Conditional Laplace  (Section 5.6, Eq 5.19)
    # ------------------------------------------------------------------

    def conditional_laplace(
        self, lam: np.ndarray, t: float, T: float, y: float
    ) -> np.ndarray:
        """Phi_{t,T}(lambda; y) = exp(-A(T-t; lambda) - B(T-t; lambda)*y).

        For the CIR clock, the conditional transform is exponential-affine
        in the current variance state y = v_t.
        Paper: Section 5.6, Eq (5.19).
        """
        # Create a temporary clock with v0 = y, same kappa/theta/xi
        tmp = CIRClock(self.kappa, self.theta, self.xi, v0=max(y, 1e-12))
        return tmp.laplace(lam, T - t)

    # ------------------------------------------------------------------
    # Monte Carlo simulation  (Section 6.1, Eqs 6.1-6.3)
    # ------------------------------------------------------------------

    def simulate_path(
        self, T: float, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate CIR variance and clock paths.

        Uses full-truncation Euler scheme (Lord et al., 2010):
            v_{t+dt} = v_t + kappa*(theta - v_t^+)*dt + xi*sqrt(v_t^+*dt)*Z
            v_t^+ = max(v_t, 0)
        Clock approximated by trapezoidal rule (Eq 6.3).

        Paper: Section 6.1, Eqs (6.1)-(6.3).

        Returns
        -------
        v_paths    : np.ndarray, shape (n_paths, n_steps+1)
        gamma_paths: np.ndarray, shape (n_paths, n_steps+1)
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        v_paths = np.empty((n_paths, n_steps + 1))
        gamma_paths = np.empty((n_paths, n_steps + 1))

        v_paths[:, 0] = self.v0
        gamma_paths[:, 0] = 0.0

        for i in range(n_steps):
            v_pos = np.maximum(v_paths[:, i], 0.0)         # v_t^+ = max(v_t, 0)
            Z = rng.standard_normal(n_paths)
            # Full-truncation Euler (Eq 6.2)
            v_paths[:, i + 1] = (
                v_paths[:, i]
                + self.kappa * (self.theta - v_pos) * dt
                + self.xi * np.sqrt(v_pos) * sqrt_dt * Z
            )
            # Trapezoidal clock integration (Eq 6.3)
            gamma_paths[:, i + 1] = (
                gamma_paths[:, i]
                + 0.5 * (v_paths[:, i] + v_paths[:, i + 1]) * dt
            )

        return v_paths, gamma_paths

    def simulate_gamma(
        self, T: float, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return only terminal Gamma_T values."""
        _, gamma_paths = self.simulate_path(T, n_paths, n_steps, rng)
        return gamma_paths[:, -1]

    def __repr__(self) -> str:
        return (
            f"CIRClock(kappa={self.kappa}, theta={self.theta}, "
            f"xi={self.xi}, v0={self.v0}, feller={self._feller_value:.4f})"
        )
