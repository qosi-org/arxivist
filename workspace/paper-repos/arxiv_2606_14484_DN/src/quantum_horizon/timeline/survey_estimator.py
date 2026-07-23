"""
Expert-survey-based CRQC arrival-year estimator.

Implements the survey-estimator side of Section 3.2 of arXiv:2606.14484:
an independent estimate of CRQC arrival year drawn from expert surveys
(Global Risk Institute / Mosca-Piani style), described in the paper as
centering near 2038-2040.

SIR reference: architecture.modules "Systemic break-year Monte-Carlo
forecast"; the exact parametric form of this estimator is not given in the
paper (ASSUMED lognormal, right-skewed to match the paper's description of
a "survey hump" with a near-term peak and a longer tail -- SIR confidence
0.45, ambiguity #1).
"""

from __future__ import annotations

import numpy as np


class SurveyBasedEstimator:
    """Samples CRQC arrival years from a distribution calibrated to the
    expert-survey mode and spread described in the paper.

    Args:
        mode_year: the survey estimator's distributional mode. The paper
            describes the survey estimator as centering "near 2038-2040",
            but for a right-skewed lognormal, median > mode, so a mode of
            2036 (config default) reproduces a median around 2038, and --
            combined with the physics estimator -- reproduces the paper's
            overall stated cumulative probabilities (by-2035 ~17%, by-2040
            ~30%, by-2050 ~60%) much better than mode=2039 does. This is an
            ASSUMED calibration tuned to match reported outputs, not an
            independently-derived value (SIR confidence 0.45).
        sigma: lognormal shape parameter controlling spread (ASSUMED default
            0.45, tuned jointly with mode_year above).
        t0: reference year for the lognormal's location parameter (2026).
    """

    def __init__(self, mode_year: float = 2036, sigma: float = 0.45, t0: float = 2026) -> None:
        self.mode_year = mode_year
        self.sigma = sigma
        self.t0 = t0

    def __repr__(self) -> str:  # noqa: D105
        return f"SurveyBasedEstimator(mode_year={self.mode_year}, sigma={self.sigma})"

    def sample_break_years(self, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n_draws survey-model CRQC arrival years.

        Uses a lognormal distribution shifted so its mode falls at
        `self.mode_year`. For a lognormal with shape sigma and scale exp(mu),
        the mode occurs at exp(mu - sigma^2); solving for mu given the target
        mode-minus-t0 gap gives the location parameter.

        Args:
            n_draws: number of Monte-Carlo samples.
            rng: NumPy random Generator for reproducibility.

        Returns:
            Array of shape [n_draws] of sampled CRQC arrival years.
        """
        target_gap = self.mode_year - self.t0
        if target_gap <= 0:
            raise ValueError("mode_year must be after t0")

        mu = np.log(target_gap) + self.sigma**2
        gaps = rng.lognormal(mean=mu, sigma=self.sigma, size=n_draws)
        return self.t0 + gaps
