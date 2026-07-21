"""
Systemic break-year Monte-Carlo forecast: blends the physics-based and
survey-based CRQC arrival estimators into the paper's headline bimodal
distribution (Section 3.2, Figure 2, arXiv:2606.14484).

    P_combined(t) = w * P_survey(t) + (1-w) * P_physics(t),  w in [0.25, 0.75]

This module mixes at the *sample* level (each Monte-Carlo draw comes from
the survey estimator with probability w, else from the physics estimator)
rather than at the density level -- a modeling choice documented in the SIR
as ambiguous (confidence 0.45), since the paper does not specify which
mixture convention it uses and the two produce different combined
distributions even with identical marginal percentiles.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from quantum_horizon.timeline.physics_estimator import PhysicsBasedEstimator
from quantum_horizon.timeline.survey_estimator import SurveyBasedEstimator


class SystemicForecastModel:
    """Blends PhysicsBasedEstimator and SurveyBasedEstimator into the
    paper's combined bimodal CRQC arrival forecast.

    Args:
        physics_estimator: a configured PhysicsBasedEstimator instance.
        survey_estimator: a configured SurveyBasedEstimator instance.
    """

    def __init__(
        self, physics_estimator: PhysicsBasedEstimator, survey_estimator: SurveyBasedEstimator
    ) -> None:
        self.physics_estimator = physics_estimator
        self.survey_estimator = survey_estimator

    def __repr__(self) -> str:  # noqa: D105
        return "SystemicForecastModel()"

    def run(self, n_draws: int = 100000, survey_weight: float = 0.5, seed: int = 0) -> Dict:
        """Run the combined Monte-Carlo forecast (Figure 2).

        Args:
            n_draws: total number of combined Monte-Carlo samples.
            survey_weight: w, the probability each draw comes from the survey
                estimator rather than the physics estimator (paper default 0.5,
                swept 0.25-0.75).
            seed: RNG seed for reproducibility.

        Returns:
            Dict with keys:
                'survey_samples': [n_draws] array from the survey estimator alone
                'physics_samples': [n_draws] array from the physics estimator alone
                'combined_samples': [n_draws] array, the actual blended forecast
                'cdf_2035', 'cdf_2040', 'cdf_2050': empirical cumulative probabilities
                'median': combined-sample median year
                'range_80pct': (10th percentile, 90th percentile) tuple
        """
        rng = np.random.default_rng(seed)

        survey_samples = self.survey_estimator.sample_break_years(n_draws, rng)
        physics_samples = self.physics_estimator.sample_break_years(n_draws, rng)

        from_survey = rng.uniform(0, 1, n_draws) < survey_weight
        combined_samples = np.where(from_survey, survey_samples, physics_samples)

        return {
            "survey_samples": survey_samples,
            "physics_samples": physics_samples,
            "combined_samples": combined_samples,
            "cdf_2035": float(np.mean(combined_samples <= 2035)),
            "cdf_2040": float(np.mean(combined_samples <= 2040)),
            "cdf_2050": float(np.mean(combined_samples <= 2050)),
            "median": float(np.median(combined_samples)),
            "range_80pct": (
                float(np.percentile(combined_samples, 10)),
                float(np.percentile(combined_samples, 90)),
            ),
        }

    def sensitivity_sweep(
        self, weights: List[float], n_draws: int = 100000, seed: int = 0
    ) -> pd.DataFrame:
        """Reproduce the paper's stated survey-weight sensitivity check:
        "at survey weights from 0.25 to 0.75 the by-2035 probability runs
        from about 8% to 24%" (Section 3.2).

        Args:
            weights: list of survey_weight values to sweep (paper: 0.25-0.75).
            n_draws: Monte-Carlo samples per weight.
            seed: RNG seed (same seed reused per weight for a fair comparison).

        Returns:
            DataFrame with columns: survey_weight, cdf_2035, cdf_2040, cdf_2050, median.
        """
        rows = []
        for w in weights:
            result = self.run(n_draws=n_draws, survey_weight=w, seed=seed)
            rows.append(
                {
                    "survey_weight": w,
                    "cdf_2035": result["cdf_2035"],
                    "cdf_2040": result["cdf_2040"],
                    "cdf_2050": result["cdf_2050"],
                    "median": result["median"],
                }
            )
        return pd.DataFrame(rows)
