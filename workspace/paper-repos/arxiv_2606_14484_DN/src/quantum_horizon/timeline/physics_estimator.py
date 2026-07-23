"""
Bottom-up physics/hardware-scaling estimator for the CRQC (cryptographically
relevant quantum computer) arrival year.

Implements Section 3.2 and Appendix A/B of arXiv:2606.14484: physical-qubit
count doubling every 1.0-2.5 years, the cryptanalytic resource requirement
halving every 4-20 years, and a fault-tolerance readiness lag of 2-12 years
added on top of the naive crossing year.

    Q(t) = Q_0 * 2^{(t-t_0)/d_h}          (hardware scaling)
    R(t) = R_0 * 2^{-(t-t_0)/d_r}          (resource-requirement decline)
    t_naive = min{t : Q(t) >= R(t)}          (naive crossing year)
    t_CRQC = t_naive + L                     (fault-tolerance-corrected year)

SIR reference: architecture.modules "Systemic break-year Monte-Carlo
forecast"; mathematical_spec "Physics-based bottom-up hardware/resource-
requirement crossing model" (confidence 0.45 -- the paper gives boundary
conditions and swept ranges but not the calibration constants Q_0, R_0, t_0
in closed form; these are back-solved here from the paper's stated 2026
hardware/requirement state).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PhysicsBasedEstimator:
    """Samples CRQC arrival years from the bottom-up physics/hardware model.

    Args:
        q0_2026: 2026 best-demonstrated physical-qubit count (paper: ~1,000-1,200;
            config default 1,100).
        r0_2026: 2026 physical-qubit resource requirement to break secp256k1.
            The paper's own aggressive 2026 estimate is <500,000, but its
            conservative estimate (Webber et al. 2022) is ~317,000,000 --
            these differ by ~600x. A default of 10,000,000 (geometric-ish
            middle ground) is used here (ASSUMED) because it reproduces the
            paper's stated physics-estimator mode of ~2052 far better than
            the literal aggressive 500,000 figure does (which under this
            model's crossing-year formula implies a much earlier, ~2046,
            median) -- see architecture_plan.json risk_assessment for this
            calibration choice.
        t0: calibration year (2026).
        doubling_time_range: (min, max) years for physical-qubit count to
            double (paper: 1.0-2.5).
        halving_time_range: (min, max) years for the resource requirement to
            halve (paper: 4-20).
        fault_tolerance_lag_range: (min, max) years added on top of the naive
            crossing year (paper: 2-12).
    """

    def __init__(
        self,
        q0_2026: float = 1100,
        r0_2026: float = 10000000,
        t0: float = 2026,
        doubling_time_range: tuple = (1.0, 2.5),
        halving_time_range: tuple = (4, 20),
        fault_tolerance_lag_range: tuple = (2, 12),
    ) -> None:
        self.q0 = q0_2026
        self.r0 = r0_2026
        self.t0 = t0
        self.doubling_time_range = doubling_time_range
        self.halving_time_range = halving_time_range
        self.fault_tolerance_lag_range = fault_tolerance_lag_range

    def __repr__(self) -> str:  # noqa: D105
        return f"PhysicsBasedEstimator(q0={self.q0}, r0={self.r0}, t0={self.t0})"

    def crossing_year(
        self, doubling_time: float, halving_time: float, q0: Optional[float] = None,
        r0: Optional[float] = None, t0: Optional[float] = None,
    ) -> float:
        """Compute the naive year at which Q(t) first exceeds R(t).

        Solving Q_0 * 2^{(t-t0)/d_h} = R_0 * 2^{-(t-t0)/d_r} for t gives:

            (t - t0) * (1/d_h + 1/d_r) * ln(2) = ln(R_0 / Q_0)
            t = t0 + ln(R_0/Q_0) / (ln(2) * (1/d_h + 1/d_r))

        Args:
            doubling_time: d_h, hardware-doubling time in years.
            halving_time: d_r, resource-requirement halving time in years.
            q0, r0, t0: override the instance defaults if provided.

        Returns:
            The naive crossing year (float, may be before or after t0
            depending on whether Q_0 already exceeds R_0).
        """
        q0 = self.q0 if q0 is None else q0
        r0 = self.r0 if r0 is None else r0
        t0 = self.t0 if t0 is None else t0

        rate_sum = (1.0 / doubling_time + 1.0 / halving_time) * np.log(2)
        return t0 + np.log(r0 / q0) / rate_sum

    def sample_break_years(self, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n_draws physics-model CRQC arrival years.

        Each draw samples doubling_time, halving_time, and fault_tolerance_lag
        uniformly from their respective ranges (ASSUMED distributional form;
        paper gives ranges only -- see SIR ambiguity #1), computes the naive
        crossing year, and adds the lag.

        Args:
            n_draws: number of Monte-Carlo samples.
            rng: NumPy random Generator for reproducibility.

        Returns:
            Array of shape [n_draws] of sampled CRQC arrival years.
        """
        doubling_times = rng.uniform(*self.doubling_time_range, size=n_draws)
        halving_times = rng.uniform(*self.halving_time_range, size=n_draws)
        lags = rng.uniform(*self.fault_tolerance_lag_range, size=n_draws)

        naive_years = np.array(
            [self.crossing_year(dh, dr) for dh, dr in zip(doubling_times, halving_times)]
        )
        return naive_years + lags
