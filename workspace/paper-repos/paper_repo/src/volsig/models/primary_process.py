"""
models/primary_process.py
─────────────────────────
Simulation of all primary process variants used as the driving noise X_t
for the signature-based volatility model.

Paper sections:
  - Section 4 (general): primary process definition
  - Section 5 / Eq. (4.2): Heston variance SDE
  - Section 6 / Eq. (6.1): fractional Brownian motion Volterra kernel variants
  - Section 6: exp(fBM) and X0·exp(fBM) variants
"""

from __future__ import annotations

import numpy as np


class PrimaryProcessSimulator:
    """
    Simulates all primary process variants described in the paper.

    Variants
    --------
    1. ``heston_variance``:  CIR/Heston variance SDE (Eq. 4.2)
       dX_t = κ(θ-X_t)dt + ν√X_t dW_t

    2. ``fbm_raw``:          Volterra fBM  (Section 6, Eq. 6.1)
       X_t = √(2H) ∫₀ᵗ (t-s)^{H-1/2} dW_s

    3. ``fbm_exp``:          exp(fBM)  (Section 6)
       X_t = exp(√(2H) ∫₀ᵗ (t-s)^{H-1/2} dW_s)

    4. ``fbm_shifted_exp``:  X0·exp(fBM)  (Section 6, best variant)
       X_t = X0·exp(√(2H) ∫₀ᵗ (t-s)^{H-1/2} dW_s)
    """

    # ── Heston variance SDE ──────────────────────────────────────────────────

    @staticmethod
    def simulate_heston_variance(
        X0: float,
        nu: float,
        kappa: float,
        theta: float,
        nMC: int,
        T_steps: int,
        dt: float,
        W: np.ndarray,  # [nMC, T_steps] pre-generated increments dW ~ N(0, dt)
    ) -> np.ndarray:
        """
        Euler-Maruyama scheme for the Heston variance SDE.  (Eq. 4.2)

        dX_t = κ(θ-X_t)dt + ν√X_t dW_t

        Uses the reflection scheme (|X|) to enforce positivity — a standard
        discretisation for CIR processes.

        Args:
            X0:      Initial variance.
            nu:      Vol-of-vol.
            kappa:   Mean-reversion speed.
            theta:   Long-run variance.
            nMC:     Number of Monte Carlo paths.
            T_steps: Number of time steps.
            dt:      Time step size.
            W:       [nMC, T_steps] pre-generated standard BM increments (√dt · Z).

        Returns:
            X: [nMC, T_steps+1] variance paths (includes X0 at index 0).
        """
        assert W.shape == (nMC, T_steps), (
            f"W must have shape [nMC={nMC}, T_steps={T_steps}], got {W.shape}"
        )
        X = np.zeros((nMC, T_steps + 1), dtype=np.float64)
        X[:, 0] = X0
        for t in range(T_steps):
            Xt = X[:, t]
            # Reflection to keep variance non-negative
            Xt_pos = np.abs(Xt)
            drift = kappa * (theta - Xt_pos) * dt
            diffusion = nu * np.sqrt(Xt_pos) * W[:, t]
            X[:, t + 1] = Xt_pos + drift + diffusion
        return X

    # ── fBM via Volterra kernel (Riemann sum approximation) ─────────────────

    @staticmethod
    def _volterra_fbm(
        H: float,
        nMC: int,
        T_steps: int,
        dt: float,
        W: np.ndarray,  # [nMC, T_steps] BM increments
    ) -> np.ndarray:
        """
        Simulate fractional BM via the Volterra kernel.  (Eq. 6.1)

        W^H_t = √(2H) ∫₀ᵗ (t-s)^{H-1/2} dW_s

        Approximated by a Riemann sum:
        W^H_{t_n} ≈ √(2H) Σ_{j=0}^{n-1} (t_n - t_j)^{H-1/2} ΔW_j

        NOTE: This is O(T_steps²) per path. For large T_steps, use the
        hybrid scheme (Bennedsen et al. 2017).  Risk R3 in architecture plan.

        Args:
            H:       Hurst parameter (0, 1).
            nMC:     Number of paths.
            T_steps: Number of time steps.
            dt:      Time step size.
            W:       [nMC, T_steps] BM increments.

        Returns:
            fBM: [nMC, T_steps+1] fBM paths (0 at index 0).
        """
        assert 0.0 < H < 1.0, f"Hurst parameter H must be in (0,1), got {H}"
        sqrt2H = np.sqrt(2.0 * H)
        times = np.arange(T_steps) * dt  # t_0, t_1, ..., t_{T-1}

        fBM = np.zeros((nMC, T_steps + 1), dtype=np.float64)
        for n in range(1, T_steps + 1):
            t_n = n * dt
            # kernel weights: (t_n - t_j)^{H-1/2} for j=0,...,n-1
            lag = t_n - times[:n]
            kernel = sqrt2H * lag ** (H - 0.5)  # shape [n]
            # dot product with BM increments: [nMC, n] @ [n] → [nMC]
            fBM[:, n] = W[:, :n] @ kernel
        return fBM

    @staticmethod
    def simulate_fbm_volterra(
        H: float,
        nMC: int,
        T_steps: int,
        dt: float,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Raw fBM primary process.  (Section 6, Eq. 6.1, variant 1)

        X_t = W^H_t  (raw Volterra fBM)

        Returns:
            X: [nMC, T_steps+1] fBM paths.
        """
        return PrimaryProcessSimulator._volterra_fbm(H, nMC, T_steps, dt, W)

    @staticmethod
    def simulate_exp_fbm(
        H: float,
        nMC: int,
        T_steps: int,
        dt: float,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Exponential fBM primary process.  (Section 6, variant 2)

        X_t = exp(W^H_t)

        Returns:
            X: [nMC, T_steps+1] positive paths.
        """
        fBM = PrimaryProcessSimulator._volterra_fbm(H, nMC, T_steps, dt, W)
        return np.exp(fBM)

    @staticmethod
    def simulate_shifted_exp_fbm(
        X0: float,
        H: float,
        nMC: int,
        T_steps: int,
        dt: float,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Shifted exponential fBM primary process.  (Section 6, variant 3 — best)

        X_t = X0 · exp(W^H_t)

        This is the best-performing variant: 17-19 min calibration time,
        loss ≈ 3.5e-4.  (Section 6)

        Args:
            X0: Initial value (arbitrary; paper uses X0=0.1).

        Returns:
            X: [nMC, T_steps+1] positive paths with X[:,0] = X0.
        """
        fBM = PrimaryProcessSimulator._volterra_fbm(H, nMC, T_steps, dt, W)
        X = X0 * np.exp(fBM)
        X[:, 0] = X0  # enforce initial condition exactly
        return X

    # ── Dispatch ─────────────────────────────────────────────────────────────

    @staticmethod
    def simulate(
        variant: str,
        W: np.ndarray,
        dt: float,
        nMC: int,
        T_steps: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Dispatch to the correct primary process simulator.

        Args:
            variant: One of 'heston_variance', 'fbm_raw', 'fbm_exp', 'fbm_shifted_exp'.
            W:       [nMC, T_steps] BM increments (dW ~ √dt · N(0,1)).
            dt:      Time step size.
            nMC:     Number of paths.
            T_steps: Number of time steps.
            **kwargs: Variant-specific parameters (e.g. X0, nu, kappa, theta, H).

        Returns:
            X: [nMC, T_steps+1] primary process paths.
        """
        dispatch = {
            "heston_variance": lambda: PrimaryProcessSimulator.simulate_heston_variance(
                X0=kwargs["X0"], nu=kwargs["nu"], kappa=kwargs["kappa"],
                theta=kwargs["theta"], nMC=nMC, T_steps=T_steps, dt=dt, W=W,
            ),
            "fbm_raw": lambda: PrimaryProcessSimulator.simulate_fbm_volterra(
                H=kwargs["H"], nMC=nMC, T_steps=T_steps, dt=dt, W=W,
            ),
            "fbm_exp": lambda: PrimaryProcessSimulator.simulate_exp_fbm(
                H=kwargs["H"], nMC=nMC, T_steps=T_steps, dt=dt, W=W,
            ),
            "fbm_shifted_exp": lambda: PrimaryProcessSimulator.simulate_shifted_exp_fbm(
                X0=kwargs["X0"], H=kwargs["H"], nMC=nMC, T_steps=T_steps, dt=dt, W=W,
            ),
        }
        if variant not in dispatch:
            raise ValueError(
                f"Unknown primary_process variant '{variant}'. "
                f"Choose from: {list(dispatch.keys())}"
            )
        return dispatch[variant]()
