"""
models/sq_ou_clock.py
=====================
Squared Ornstein-Uhlenbeck stochastic clock.

The latent factor Y_t follows an OU process:
    dY_t = -alpha*Y_t*dt + sigma*dZ_t

The variance and clock are:
    v_t = Y_t²,   Gamma_T = integral_0^T Y_s² ds

The Laplace transform is exponential-quadratic:
    Phi_T(lambda) = exp(-A(T;lambda) - B(T;lambda)*Y0²)
where A, B satisfy the Riccati ODE (4.10)-(4.11).

Paper: Section 4.2, Eqs (4.7)-(4.11) and (6.4)-(6.5).
arXiv: 2605.06677v1
"""

import numpy as np
from scipy.integrate import solve_ivp
from .base_clock import BaseClock


class SquaredOUClock(BaseClock):
    """Squared OU stochastic clock (Stein-Stein / Schöbel-Zhu type clock).

    Attributes
    ----------
    alpha : float  Mean-reversion speed (α > 0)
    sigma : float  OU diffusion coefficient (σ > 0)
    Y0    : float  Initial OU factor value (any real number)

    Paper reference: Section 4.2
    """

    def __init__(self, alpha: float, sigma: float, Y0: float) -> None:
        """
        Parameters
        ----------
        alpha : float  Mean-reversion speed α > 0
        sigma : float  OU diffusion coefficient σ > 0
        Y0    : float  Initial OU factor Y_0 ∈ ℝ (v0 = Y0²)
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.alpha = alpha
        self.sigma = sigma
        self.Y0 = Y0

    @property
    def v0(self) -> float:
        """Initial variance v0 = Y0²."""
        return self.Y0**2

    # ------------------------------------------------------------------
    # Riccati ODE  (Section 4.2, Eqs 4.10-4.11)
    # ------------------------------------------------------------------

    def _riccati_rhs(self, t: float, y: np.ndarray, lam: float) -> list:
        """RHS of Riccati ODE for squared OU clock.

        State: y = [B(t), A(t)]

        dB/dt = 2*sigma²*B² - 2*alpha*B + lambda   (Eq 4.10)
        dA/dt = sigma²*B                             (Eq 4.11)
        """
        B, A = y
        dB = 2.0 * self.sigma**2 * B**2 - 2.0 * self.alpha * B + lam
        dA = self.sigma**2 * B
        return [dB, dA]

    def _solve_riccati(self, lam: float, T: float) -> tuple[float, float]:
        """Solve Riccati for single lambda; return (A(T), B(T))."""
        sol = solve_ivp(
            self._riccati_rhs,
            t_span=(0.0, T),
            y0=[0.0, 0.0],
            args=(lam,),
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(
                f"Squared-OU Riccati failed for lambda={lam}, T={T}: {sol.message}"
            )
        return float(sol.y[1, -1]), float(sol.y[0, -1])

    # ------------------------------------------------------------------
    # Laplace transform  (Section 4.2, Eq 4.9)
    # ------------------------------------------------------------------

    def laplace(self, lam: np.ndarray, T: float) -> np.ndarray:
        """Phi_T(lambda) = exp(-A(T;lambda) - B(T;lambda)*Y0²).

        Paper: Section 4.2, Eq (4.9).
        """
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        result = np.empty_like(lam)
        for i, li in enumerate(lam.ravel()):
            A_T, B_T = self._solve_riccati(li, T)
            log_phi = -A_T - B_T * self.Y0**2
            result.ravel()[i] = np.exp(log_phi)
        return result.reshape(lam.shape)

    def log_laplace(self, lam: np.ndarray, T: float) -> np.ndarray:
        """log Phi_T(lambda) = -A(T;lambda) - B(T;lambda)*Y0²."""
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        result = np.empty_like(lam)
        for i, li in enumerate(lam.ravel()):
            A_T, B_T = self._solve_riccati(li, T)
            result.ravel()[i] = -A_T - B_T * self.Y0**2
        return result.reshape(lam.shape)

    def d_laplace_dy(self, lam: np.ndarray, T: float) -> np.ndarray:
        """d/dY0 Phi_T(lambda) = -2*B(T;lambda)*Y0 * Phi_T(lambda).

        Paper: Analogous to Eq (6.8) for squared OU.
        From d/dY0 exp(-A - B*Y0²) = -2*B*Y0 * exp(-A - B*Y0²).
        """
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        phi = self.laplace(lam, T)
        B_vals = np.empty_like(lam)
        for i, li in enumerate(lam.ravel()):
            _, B_T = self._solve_riccati(li, T)
            B_vals.ravel()[i] = B_T
        return -2.0 * B_vals * self.Y0 * phi

    def conditional_laplace(
        self, lam: np.ndarray, t: float, T: float, y: float
    ) -> np.ndarray:
        """Phi_{t,T}(lambda; y) with Y_t = y (remaining horizon T-t).

        Paper: Section 5.6, Eq (5.19).
        """
        tmp = SquaredOUClock(self.alpha, self.sigma, Y0=y)
        return tmp.laplace(lam, T - t)

    # ------------------------------------------------------------------
    # Monte Carlo simulation  (Section 6.1, Eqs 6.4-6.5)
    # ------------------------------------------------------------------

    def simulate_path(
        self, T: float, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate squared-OU variance and clock paths.

        Uses exact OU step update (Glasserman, 2003):
            nu_{t+dt} = nu_t * exp(-alpha*dt)
                        + sigma * sqrt((1 - exp(-2*alpha*dt)) / (2*alpha)) * Z

        Paper: Section 6.1, Eqs (6.4)-(6.5).

        Returns
        -------
        v_paths    : np.ndarray, shape (n_paths, n_steps+1)  [= nu_paths²]
        gamma_paths: np.ndarray, shape (n_paths, n_steps+1)
        """
        dt = T / n_steps

        # Exact OU conditional std (Eq 6.5)
        exp_adt = np.exp(-self.alpha * dt)
        cond_std = self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.alpha * dt))
                                        / (2.0 * self.alpha))

        nu_paths = np.empty((n_paths, n_steps + 1))  # OU factor
        gamma_paths = np.empty((n_paths, n_steps + 1))

        nu_paths[:, 0] = self.Y0
        gamma_paths[:, 0] = 0.0

        for i in range(n_steps):
            Z = rng.standard_normal(n_paths)
            # Exact step (Eq 6.5)
            nu_paths[:, i + 1] = exp_adt * nu_paths[:, i] + cond_std * Z
            # v_t = nu_t² ; trapezoidal clock integration
            v_curr = nu_paths[:, i] ** 2
            v_next = nu_paths[:, i + 1] ** 2
            gamma_paths[:, i + 1] = (
                gamma_paths[:, i] + 0.5 * (v_curr + v_next) * dt
            )

        v_paths = nu_paths**2
        return v_paths, gamma_paths

    def simulate_gamma(
        self, T: float, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        _, gamma_paths = self.simulate_path(T, n_paths, n_steps, rng)
        return gamma_paths[:, -1]

    def __repr__(self) -> str:
        return (
            f"SquaredOUClock(alpha={self.alpha}, sigma={self.sigma}, "
            f"Y0={self.Y0}, v0={self.v0:.6f})"
        )
