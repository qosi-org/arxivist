"""
Hardware contrast calibration and readout-error mitigation.

Implements Section 3.2.2 ("Noise-aware amplitude estimation algorithm
validation") of arXiv:2607.12990: fitting the exponential contrast-decay
model from a Grover-power sweep on real hardware, readout-error mitigation,
and building the "hardware-replay" statistical model used to generate R=300
independent adaptive trajectories without running every trajectory live on
the QPU.

Key equations implemented:
  - contrast sample: c_hat_k = (p_hat_k - b) / (q_k - b)
  - exponential fit: log(c_hat_k) = beta0 + beta1*K, tau_c = -1/beta1
  - readout mitigation: p_mit = clip((p_raw - r0) / (r1 - r0), [0,1])
  - hardware-replay: n_1(k) ~ Binomial(n_shots, p_hat_hw(1|k))

SIR reference: mathematical_spec "CABIQAE noise-aware observation model",
architecture.modules "CABIQAE classical estimator" (contrast calibration
feeds directly into its noise model).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


class ContrastCalibrator:
    """Fits the exponential Grover-contrast decay model and builds the
    hardware-replay measurement model used by CABIQAE/BIQAE/BAE.

    Args:
        b: asymptotic baseline probability (fixed prior to fitting, e.g. 0.5
            for the balanced validation circuit or 0.15 for the full CVA
            oracle, matching the paper's per-circuit calibration in Tables
            17-18).
    """

    def __init__(self, b: float) -> None:
        self.b = b

    def __repr__(self) -> str:  # noqa: D105
        return f"ContrastCalibrator(b={self.b})"

    def readout_mitigate(
        self, raw_probs: np.ndarray, r0: float, r1: float
    ) -> np.ndarray:
        """Affine readout-error correction (Section 3.2.2).

        p_mit = clip((p_raw - r0) / (r1 - r0), [0, 1]),
        where r0 = P(obs=1|prep=0), r1 = P(obs=1|prep=1).

        Args:
            raw_probs: raw measured success probabilities per Grover power.
            r0, r1: calibration-circuit measured probabilities.

        Returns:
            Mitigated probabilities, clipped to [0, 1].
        """
        delta_read = r1 - r0
        if abs(delta_read) < 1e-9:
            raise ValueError("Readout calibration is unstable (Delta_read ~ 0)")
        mitigated = (raw_probs - r0) / delta_read
        return np.clip(mitigated, 0.0, 1.0)

    def fit_contrast_model(
        self,
        k_values: np.ndarray,
        observed_probs: np.ndarray,
        ideal_theta: float,
        exclude_unstable: bool = True,
        stability_threshold: float = 1e-3,
    ) -> Tuple[float, float, float]:
        """Fit c0 and tau_c from a Grover-power calibration sweep.

        c_hat_k = (p_hat_k - b) / (q_k - b),   q_k = sin^2((2k+1) * ideal_theta)
        log(c_hat_k) = beta0 + beta1 * K,      K = 2k+1
        tau_c = -1 / beta1   (requires beta1 < 0; otherwise fit is rejected
        and the ideal observation model should be used instead, per the
        paper's stated fallback in Section 3.2.2)

        Args:
            k_values: array of Grover powers used in the calibration sweep.
            observed_probs: measured (readout-mitigated) success probabilities
                at each k.
            ideal_theta: arcsin(sqrt(a_true)) for the known calibration
                amplitude (oracle-assisted calibration, per the paper's
                explicit acknowledgement in Section 3.2.2 footnote 13).
            exclude_unstable: if True, drop points with unstable/non-physical
                contrast values before fitting.
            stability_threshold: minimum |q_k - b| to keep a calibration
                point (avoids division-by-near-zero denominators).

        Returns:
            (c0, tau_c, r_squared). If no valid negative-slope fit is found,
            returns (1.0, np.inf, 0.0) as a signal to fall back to the ideal
            observation model (matching the paper's stated behaviour).
        """
        K = 2 * k_values + 1
        q_k = np.sin(K * ideal_theta) ** 2
        denom = q_k - self.b

        valid = np.ones_like(denom, dtype=bool)
        if exclude_unstable:
            valid &= np.abs(denom) > stability_threshold

        c_hat = np.full_like(denom, np.nan)
        c_hat[valid] = (observed_probs[valid] - self.b) / denom[valid]
        valid &= (c_hat > 0) & np.isfinite(c_hat)

        if valid.sum() < 2:
            return 1.0, np.inf, 0.0

        K_valid = K[valid]
        log_c = np.log(c_hat[valid])

        beta1, beta0 = np.polyfit(K_valid, log_c, 1)
        if beta1 >= 0:
            # No valid negative-slope fit -- reject calibration (Section 3.2.2)
            return 1.0, np.inf, 0.0

        tau_c = -1.0 / beta1
        c0 = float(np.exp(beta0))

        pred = beta0 + beta1 * K_valid
        ss_res = np.sum((log_c - pred) ** 2)
        ss_tot = np.sum((log_c - log_c.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return c0, tau_c, float(r_squared)

    def build_hardware_replay_model(
        self, hardware_probs: Dict[int, float], rng: Optional[np.random.Generator] = None
    ) -> Callable[[int, int], Tuple[int, int]]:
        """Build a circuit_executor callable that replays hardware-measured
        Grover-power responses via Binomial sampling (Section 3.2.2):

            n_1(k) ~ Binomial(n_shots, p_hat_hw(1|k))

        This lets CABIQAE/BIQAE/BAE run R=300 independent adaptive
        trajectories from a single set of real hardware measurements,
        without executing every trajectory live on the QPU.

        Args:
            hardware_probs: dict mapping Grover power k -> measured
                (readout-mitigated) success probability p_hat_hw(1|k).
            rng: optional NumPy random generator.

        Returns:
            A CircuitExecutor-compatible callable (k, n_shots) ->
            (n_success, n_shots). Raises KeyError if a requested k was not
            measured on hardware.
        """
        rng = rng or np.random.default_rng()

        def executor(k: int, n_shots: int) -> Tuple[int, int]:
            if k not in hardware_probs:
                raise KeyError(
                    f"Grover power k={k} was not measured in the hardware "
                    f"calibration sweep; available k values: "
                    f"{sorted(hardware_probs.keys())}"
                )
            p = hardware_probs[k]
            n_success = int(rng.binomial(n_shots, p))
            return n_success, n_shots

        return executor
