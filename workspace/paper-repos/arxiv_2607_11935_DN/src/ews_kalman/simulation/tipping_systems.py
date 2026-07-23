"""
Six simulated dynamical systems with known tipping points, used to validate
that beta leads AR(1) specifically when the transition involves coupling
degradation (Section 2.4, Figure 3, Table 3 of arXiv:2607.11935).

The paper gives only qualitative governing equations and bifurcation-
parameter endpoints for each system (noise magnitudes, exact forcing
schedules, and simulation lengths are not specified numerically -- SIR
ambiguity #2, implementation_assumptions[3], confidence ~0.4). Illustrative
default parameters are used here and exposed as function arguments /
config.yaml fields so they can be tuned.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class TippingSystemSimulator:
    """Generators for the six simulated tipping-point systems."""

    def __repr__(self) -> str:  # noqa: D105
        return "TippingSystemSimulator()"

    def fold_bifurcation(
        self,
        n_steps: int = 250,
        tipping_t: int = 200,
        noise_std: float = 0.05,
        dt: float = 1.0,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Fold bifurcation: dx/dt = r(t) - x^2 + eta, r -> 0 at t=tipping_t.

        r(t) ramps linearly from a positive value down to (and through) 0 at
        tipping_t, causing x to blow up as the fold is crossed (Section 2.4).
        """
        rng = np.random.default_rng(seed)
        r = np.linspace(1.0, -0.2, n_steps)
        # shift so r(tipping_t) == 0
        shift = -r[tipping_t]
        r = r + shift

        x = np.zeros(n_steps)
        x[0] = np.sqrt(max(r[0], 0.01))
        for k in range(1, n_steps):
            dx = (r[k - 1] - x[k - 1] ** 2) * dt + rng.normal(0, noise_std) * np.sqrt(dt)
            x[k] = x[k - 1] + dx
            if not np.isfinite(x[k]) or abs(x[k]) > 1e10:
                x[k:] = x[k]
                break

        return {"x": x, "r": r, "tipping_index": tipping_t}

    def beta_step_change(
        self,
        n_steps: int = 300,
        tipping_t: int = 200,
        beta_before: float = 0.8,
        beta_after: float = 0.2,
        noise_std: float = 0.1,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Beta step change: y = beta(t)*x + eps, beta: beta_before -> beta_after at tipping_t."""
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n_steps).cumsum() * 0.1 + 10  # slowly-varying positive driver
        beta_true = np.where(np.arange(n_steps) < tipping_t, beta_before, beta_after)
        y = beta_true * x + rng.normal(0, noise_std, n_steps)

        return {"x": x, "y": y, "beta_true": beta_true, "tipping_index": tipping_t}

    def beta_linear_decay(
        self,
        n_steps: int = 300,
        decay_end: int = 250,
        beta_start: float = 1.0,
        beta_end: float = 0.2,
        noise_std: float = 0.1,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Beta linear decay: beta decays linearly from beta_start to beta_end
        over [0, decay_end], then holds constant (coupling degradation)."""
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n_steps).cumsum() * 0.1 + 10

        beta_true = np.empty(n_steps)
        ramp_len = min(decay_end, n_steps)
        beta_true[:ramp_len] = np.linspace(beta_start, beta_end, ramp_len)
        if n_steps > ramp_len:
            beta_true[ramp_len:] = beta_end

        y = beta_true * x + rng.normal(0, noise_std, n_steps)

        return {"x": x, "y": y, "beta_true": beta_true, "tipping_index": decay_end}

    def logistic_map(
        self,
        n_steps: int = 400,
        tipping_t: int = 133,
        r_start: float = 2.5,
        r_end: float = 4.0,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Logistic map: x_{n+1} = r*x_n*(1-x_n), r: r_start -> r_end (Feigenbaum cascade)."""
        rng = np.random.default_rng(seed)
        r = np.linspace(r_start, r_end, n_steps)
        x = np.zeros(n_steps)
        x[0] = 0.5 + rng.normal(0, 1e-3)
        for k in range(1, n_steps):
            x[k] = r[k - 1] * x[k - 1] * (1 - x[k - 1])
            x[k] = np.clip(x[k], 1e-9, 1 - 1e-9)

        return {"x": x, "r": r, "tipping_index": tipping_t}

    def stommel_amoc(
        self,
        n_steps: int = 300,
        tipping_t: int = 200,
        freshwater_forcing_rate: float = 0.005,
        noise_std: float = 0.02,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Simplified Stommel (1961) 2-box thermohaline circulation model
        with linearly increasing freshwater forcing, causing an AMOC
        collapse (overturning strength psi -> 0) near tipping_t.

        dT/dt = -T - |psi|*T
        dS/dt = F(t) - |psi|*S
        psi = T - S  (density-difference-driven overturning strength)
        """
        rng = np.random.default_rng(seed)
        dt = 1.0
        T = np.zeros(n_steps)
        S = np.zeros(n_steps)
        psi = np.zeros(n_steps)
        T[0], S[0] = 1.0, 0.5
        psi[0] = T[0] - S[0]

        for k in range(1, n_steps):
            F_forcing = freshwater_forcing_rate * k
            dT = (-T[k - 1] - abs(psi[k - 1]) * T[k - 1]) * dt + rng.normal(0, noise_std) * np.sqrt(dt)
            dS = (F_forcing - abs(psi[k - 1]) * S[k - 1]) * dt + rng.normal(0, noise_std) * np.sqrt(dt)
            T[k] = T[k - 1] + dT
            S[k] = S[k - 1] + dS
            psi[k] = T[k] - S[k]

        return {"T": T, "S": S, "psi": psi, "tipping_index": tipping_t}

    def critical_slowing_down(
        self,
        n_steps: int = 300,
        tipping_t: int = 250,
        lambda_start: float = 1.0,
        lambda_end: float = 0.0,
        noise_std: float = 0.1,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Critical slowing down: dx/dt = -lambda(t)*x + eta, lambda -> 0
        (the relaxation rate vanishes, so recovery from perturbations slows)."""
        rng = np.random.default_rng(seed)
        dt = 1.0
        lam = np.linspace(lambda_start, lambda_end, n_steps)
        x = np.zeros(n_steps)
        x[0] = rng.normal(0, 0.1)
        for k in range(1, n_steps):
            dx = -lam[k - 1] * x[k - 1] * dt + rng.normal(0, noise_std) * np.sqrt(dt)
            x[k] = x[k - 1] + dx

        return {"x": x, "lambda": lam, "tipping_index": tipping_t}
