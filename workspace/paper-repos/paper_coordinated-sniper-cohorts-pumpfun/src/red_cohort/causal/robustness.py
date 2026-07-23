"""
causal/robustness.py
---------------------
Robustness checks from Section 6.5 and Appendix B.2–B.3.

Checks implemented:
1. Top-k exclusion (Appendix B.2): remove top-3 cohorts by score.
2. Tier stratification (Appendix B.3 / Section 6.5): estimate lift per tier.

Paper: Kamat (2026), Section 6.5.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from red_cohort.causal.estimator import LiftEstimator
from red_cohort.utils.config import CausalConfig


class RobustnessChecker:
    """
    Runs robustness checks on the causal lift estimates.

    Paper reference:
        Section 6.5 — "Top-3 cohort exclusion" and "Tier stratification".
        Appendix B.2 — top-k exclusion yields +128.8% (vs +132.3% baseline).
        Appendix B.3 — tier lifts: Standard +122.8%, High +131.4%, Premium +79.5%.

    Args:
        config: CausalConfig (for bootstrap settings).
    """

    def __init__(self, config: CausalConfig) -> None:
        self.cfg = config
        self.estimator = LiftEstimator(config)

    def top_k_exclusion(
        self,
        cohorts_df: pd.DataFrame,
        k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return cohorts_df with the top-k scoring cohorts removed.

        Paper: removes COH-0001, COH-0002, COH-0003 (Appendix B.2).
        Remaining: 1,009 cohorts touching 5,869 launches, lift = +128.8%.

        Args:
            cohorts_df: Full cohort catalogue sorted by score DESC.
            k: Number of top cohorts to exclude (default from config: 3).

        Returns:
            Filtered cohorts_df with top-k rows removed.
        """
        k = k if k is not None else self.cfg.top_k_exclusion
        return cohorts_df.iloc[k:].copy()

    def tier_stratification(
        self,
        cohorts_df: pd.DataFrame,
        treated_df: pd.DataFrame,
        control_df: pd.DataFrame,
        intra_index: pd.DataFrame,
        outcome_col: str = "first_30min_buyer_count",
    ) -> pd.DataFrame:
        """
        Compute lift estimates separately for Standard, High, and Premium tiers.

        Paper (Appendix B.3 / Section 6.5):
            Standard: n_treated=3,747, buyer_lift=+122.8%, SOL_lift=+136.0%
            High:     n_treated=1,688, buyer_lift=+131.4%, SOL_lift=+125.6%
            Premium:  n_treated=540,   buyer_lift=+79.5%,  SOL_lift=+84.5%

        Note: Non-monotone pattern (Premium < Standard) is a key diagnostic
        result interpreted as evidence against a coordination-causes-flow story.

        Args:
            cohorts_df: Cohort catalogue with 'tier' column.
            treated_df: Treated group outcomes (with 'mint' column for join).
            control_df: Control group outcomes.
            intra_index: For resolving mint → tier mapping.
            outcome_col: Primary outcome column to estimate.

        Returns:
            DataFrame with tier, n_treated, point_estimate_pct, ci_lower_pct, ci_upper_pct.
        """
        # Build mint → tier mapping
        # A mint's tier is the highest tier of any touching cohort
        tier_order = {"premium": 3, "high": 2, "standard": 1}

        all_cohort_wallets: dict = {}
        for _, row in cohorts_df.iterrows():
            for w in row["wallets"]:
                tier_rank = tier_order.get(row.get("tier", "standard"), 1)
                existing = all_cohort_wallets.get(w, (0, "standard"))
                if tier_rank > existing[0]:
                    all_cohort_wallets[w] = (tier_rank, row.get("tier", "standard"))

        rows = []
        for tier_name in ["standard", "high", "premium"]:
            # Wallets belonging to this tier or above
            tier_wallets = {w for w, (rank, t) in all_cohort_wallets.items() if t == tier_name}

            # Find mints touched by wallets of exactly this tier
            tier_rows = intra_index[intra_index["wallet"].isin(tier_wallets)]
            tier_mints = set(tier_rows["mint"].unique())

            tier_treated = treated_df[treated_df["mint"].isin(tier_mints)]

            if len(tier_treated) < 10:
                continue

            result = self.estimator.estimate(tier_treated, control_df, outcome_col)
            result["tier"] = tier_name
            rows.append(result)

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"RobustnessChecker(top_k={self.cfg.top_k_exclusion})"
