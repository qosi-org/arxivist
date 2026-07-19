"""
causal/sample.py
-----------------
Constructs treated and control samples for the 3:1 random-matched causal design.

Paper: Kamat (2026), Section 6.1 (sample construction) and Section 6.2 (outcomes).
"""
from __future__ import annotations

import random
from typing import List, Tuple

import pandas as pd

from red_cohort.utils.config import CausalConfig


class CausalSampleBuilder:
    """
    Builds treated and control mint sets and attaches first-30-minute outcome variables.

    Treated: launches where >=touch_threshold cohort wallets appear in first-10 buyers.
    Control: random sample of untouched launches drawn at control_ratio:1 vs treated.

    Paper reference:
        Section 6.1 — "cohort-touched when at least two cohort wallets appear among the
        first ten buyers... 3:1 treated-to-control ratio."
        Section 6.2 — Outcome variables: first_30min_buyer_count, first_30min_sol_inflow,
        total_buyer_count.

    Args:
        config: CausalConfig with window_minutes, control_ratio, random_seed.
    """

    def __init__(self, config: CausalConfig) -> None:
        self.cfg = config

    def build_treated(
        self,
        cohorts_df: pd.DataFrame,
        intra_index: pd.DataFrame,
        touch_threshold: int = 2,
    ) -> List[str]:
        """
        Identify launches where >=touch_threshold cohort wallets appear in first-10 buyers.

        Paper: strict threshold of >=2 cohort wallets (Section 6.1).
        Returns 5,411 mints in the paper corpus.

        Args:
            cohorts_df: Scored cohort catalogue from DetectionPipeline.
            intra_index: Per-launch first-buyer index from IntraLaunchExtractor.
            touch_threshold: Minimum number of cohort wallets required (default: 2).
        """
        # Collect all cohort wallet addresses
        all_cohort_wallets: set = set()
        for wallets in cohorts_df["wallets"]:
            all_cohort_wallets.update(wallets)

        # For each mint, count how many distinct cohort wallets appear in first-10
        cohort_mask = intra_index["wallet"].isin(all_cohort_wallets)
        cohort_rows = intra_index[cohort_mask]

        mint_cohort_counts = (
            cohort_rows
            .groupby("mint")["wallet"]
            .nunique()
        )
        treated_mints = mint_cohort_counts[mint_cohort_counts >= touch_threshold].index.tolist()
        return treated_mints

    def build_control(
        self,
        all_mints: List[str],
        treated_mints: List[str],
        ratio: int = 3,
        seed: int = 42,
    ) -> List[str]:
        """
        Draw a random control sample of untouched launches at control_ratio:1 vs treated.

        Paper: "We draw a random control sample of 16,233 untouched launches
        (uniform, without replacement, seed = 42), yielding a 3:1 treated-to-control ratio."

        Args:
            all_mints: All qualifying mint addresses.
            treated_mints: Mint addresses in the treated group.
            ratio: Control-to-treated ratio (default: 3).
            seed: Random seed (paper: 42).
        """
        treated_set = set(treated_mints)
        untouched = [m for m in all_mints if m not in treated_set]

        n_control = min(ratio * len(treated_mints), len(untouched))
        rng = random.Random(seed)
        control_mints = rng.sample(untouched, n_control)
        return control_mints

    def attach_outcomes(
        self,
        mints: List[str],
        buyers_df: pd.DataFrame,
        launches_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute outcome variables for each mint in *mints*.

        Outcomes (Section 6.2):
            1. first_30min_buyer_count  — distinct buyers within window_minutes of launch
            2. first_30min_sol_inflow   — sum of sol_in within window_minutes
            3. total_buyer_count        — total buyers across full corpus window

        Args:
            mints: List of mint addresses to evaluate.
            buyers_df: Full buyer events DataFrame.
            launches_df: Launch metadata DataFrame (for created_timestamp).

        Returns:
            DataFrame with columns: {mint, first_30min_buyer_count,
            first_30min_sol_inflow, total_buyer_count}.
        """
        window_seconds = self.cfg.window_minutes * 60

        # Index launches by mint for fast lookup
        launch_ts = launches_df.set_index("mint")["created_timestamp"].to_dict()

        mint_set = set(mints)
        relevant_buyers = buyers_df[buyers_df["mint"].isin(mint_set)].copy()

        rows = []
        for mint, group in relevant_buyers.groupby("mint"):
            created = launch_ts.get(mint)
            if created is None:
                continue

            total_buyers = len(group)

            within_window = group[group["blockTime"] <= created + window_seconds]
            first_30min_buyers = len(within_window)
            first_30min_sol = within_window["sol_in"].sum()

            rows.append({
                "mint": mint,
                "first_30min_buyer_count": first_30min_buyers,
                "first_30min_sol_inflow": float(first_30min_sol),
                "total_buyer_count": total_buyers,
            })

        return pd.DataFrame(rows)

    def build(
        self,
        cohorts_df: pd.DataFrame,
        intra_index: pd.DataFrame,
        buyers_df: pd.DataFrame,
        launches_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full sample construction: returns (treated_df, control_df) with outcomes attached.

        Returns:
            treated_df: ~5,411 rows (paper value at default settings).
            control_df: ~16,233 rows.
        """
        treated_mints = self.build_treated(
            cohorts_df, intra_index, touch_threshold=self.cfg.window_minutes
        )
        # Correct: use touch_threshold_causal, not window_minutes
        treated_mints = self.build_treated(
            cohorts_df, intra_index, touch_threshold=2
        )

        all_mints = intra_index["mint"].unique().tolist()
        control_mints = self.build_control(
            all_mints, treated_mints,
            ratio=self.cfg.control_ratio,
            seed=self.cfg.random_seed,
        )

        treated_df = self.attach_outcomes(treated_mints, buyers_df, launches_df)
        control_df = self.attach_outcomes(control_mints, buyers_df, launches_df)

        return treated_df, control_df

    def __repr__(self) -> str:
        return f"CausalSampleBuilder(window_minutes={self.cfg.window_minutes})"
