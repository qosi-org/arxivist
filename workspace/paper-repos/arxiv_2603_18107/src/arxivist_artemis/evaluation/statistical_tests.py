"""
evaluation/statistical_tests.py
==================================
Wilcoxon signed-rank significance testing across seeds (Discussion section:
"ARTEMIS's improvement over the best baseline on DSLOB and Time-IMM is
statistically significant (p<0.01, Wilcoxon signed-rank test)").

NOTE: the paper does not fully specify which metric the test was run on, nor
which exact baseline was used as "the best baseline" per dataset (it likely
varies: e.g. best DirAcc baseline on DSLOB vs. best DirAcc baseline on
Time-IMM). This module implements the test generically; the caller must
supply the specific paired seed-level scores to compare.
"""

from __future__ import annotations

from typing import List

from scipy.stats import wilcoxon


class SignificanceTester:
    """Stateless wrapper around scipy's Wilcoxon signed-rank test."""

    @staticmethod
    def wilcoxon_vs_baseline(artemis_scores: List[float], baseline_scores: List[float]) -> dict:
        """
        Args:
            artemis_scores: per-seed metric values for ARTEMIS (e.g. 5 seeds).
            baseline_scores: per-seed metric values for the comparison baseline,
                same length and seed-alignment as artemis_scores.
        Returns:
            {'statistic': float, 'p_value': float}
        """
        assert len(artemis_scores) == len(baseline_scores), (
            "artemis_scores and baseline_scores must have matching per-seed length "
            f"(got {len(artemis_scores)} vs {len(baseline_scores)})"
        )
        try:
            statistic, p_value = wilcoxon(artemis_scores, baseline_scores)
        except ValueError:
            # wilcoxon raises if all differences are zero; report non-significant.
            return {"statistic": 0.0, "p_value": 1.0}
        return {"statistic": float(statistic), "p_value": float(p_value)}
