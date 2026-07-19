"""
analysis/descriptive.py
------------------------
Produces all descriptive statistics from Section 5 (Tables 1-3 and Lorenz curve data).

Paper: Kamat (2026), Section 5.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DescriptiveAnalyzer:
    """
    Computes summary statistics matching Tables 1-3 and Figure 2 (Lorenz curve).

    Paper reference: Section 5 — 1,012 cohorts, 2,965 unique wallets,
    median size 2, max score 430.44, Lorenz concentration analysis.
    """

    def top_k_cohorts(
        self,
        cohorts_df: pd.DataFrame,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Table 1: Top-k cohorts by score.
        Paper: COH-0001 has score=430.44, 9 wallets, 42 launches, mean_rank=2.29.
        """
        display_cols = [
            "cohort_id", "size", "n_launches_hit",
            "mean_first_rank", "total_sol", "score", "tier",
        ]
        available = [c for c in display_cols if c in cohorts_df.columns]
        return cohorts_df.nlargest(k, "score")[available].reset_index(drop=True)

    def size_distribution(self, cohorts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Table 2: Cohort size distribution.
        Paper: size=2 → 571 cohorts (56.4%), size=3 → 229 (22.6%), ..., size=12 → 3.
        """
        counts = cohorts_df["size"].value_counts().sort_index()
        df = pd.DataFrame({"size": counts.index, "n_cohorts": counts.values})
        df["pct"] = (df["n_cohorts"] / df["n_cohorts"].sum() * 100).round(1)
        return df

    def headline_stats(self, cohorts_df: pd.DataFrame) -> Dict:
        """
        Table 3: Headline descriptive statistics.
        Paper values for reference:
            total_cohorts=1012, unique_wallets=2965,
            median_size=2, mean_size=2.93, max_size=12,
            median_launches=5, mean_launches=6.23, max_launches=42,
            median_score=52.8, max_score=430.44,
            high_tier=153, premium_tier=22.
        """
        all_wallets = set()
        for wallets in cohorts_df["wallets"]:
            all_wallets.update(wallets)

        return {
            "total_cohorts": len(cohorts_df),
            "unique_cohort_wallets": len(all_wallets),
            "median_size": float(cohorts_df["size"].median()),
            "mean_size": round(float(cohorts_df["size"].mean()), 2),
            "max_size": int(cohorts_df["size"].max()),
            "median_launches_hit": float(cohorts_df["n_launches_hit"].median()),
            "mean_launches_hit": round(float(cohorts_df["n_launches_hit"].mean()), 2),
            "max_launches_hit": int(cohorts_df["n_launches_hit"].max()),
            "median_score": round(float(cohorts_df["score"].median()), 2),
            "max_score": round(float(cohorts_df["score"].max()), 2),
            "high_tier_cohorts": int((cohorts_df["tier"].isin(["high", "premium"])).sum())
            if "tier" in cohorts_df.columns else "N/A",
            "premium_tier_cohorts": int((cohorts_df["tier"] == "premium").sum())
            if "tier" in cohorts_df.columns else "N/A",
        }

    def lorenz_data(
        self,
        cohorts_df: pd.DataFrame,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Lorenz curve data for Figure 2 (cohort activity concentration).

        Cohorts sorted descending by n_launches_hit.
        Paper: top 10 cohorts (≈1%) account for >9% of all cohort-touch events;
               top 100 (≈10%) account for >35%.

        Returns:
            (x_vals, y_vals) where x is cumulative share of cohorts and
            y is cumulative share of launches hit.
        """
        sorted_launches = cohorts_df["n_launches_hit"].sort_values(ascending=False).values
        n = len(sorted_launches)
        total = sorted_launches.sum()

        cumsum = np.cumsum(sorted_launches)
        x_vals = [(i + 1) / n for i in range(n)]
        y_vals = [c / total for c in cumsum]

        return x_vals, y_vals

    def __repr__(self) -> str:
        return "DescriptiveAnalyzer()"
