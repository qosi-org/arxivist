"""
detection/scorer.py
--------------------
Implements EQ1 (Cohort Score Function) and tier classification.

Paper: Kamat (2026), Section 4.2 — scoring formula and tier thresholds.

EQ1:
    score(C) = 10 × Σ_L 1{C touches L}
             + 5 / mean_first_rank(C)
             + √( Σ_L SOL_committed(C, L) )

where:
    C                   = cohort (set of wallet addresses)
    1{C touches L}      = 1 if >=touch_threshold wallets in C appear in first-10 of L
    mean_first_rank(C)  = mean over touched launches of the min rank attained by any C wallet
    SOL_committed(C, L) = sum of sol_in for all C wallets in launch L
"""
from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional

import pandas as pd


class CohortScorer:
    """
    Computes EQ1 composite score for each candidate cohort.

    Args:
        touch_threshold_score: Number of cohort wallets that must appear in a launch
            for it to count as "touched" in the score formula.
            # WARNING: low-confidence implementation (SIR confidence 0.68)
            # TODO: verify with paper author whether score uses >=1 (loose)
            #       or >=2 (strict, matching causal analysis) — default=1
        tau: Score threshold below which cohorts are discarded.
            # WARNING: low-confidence implementation (SIR confidence 0.55)
            # TODO: exact tau value not disclosed. Default=40.0 is below reported
            #       median score of 52.8. Use calibrate() to back-solve.

    Paper reference: Section 4.2, EQ1.
    """

    def __init__(
        self,
        touch_threshold_score: int = 1,
        tau: float = 40.0,
    ) -> None:
        # WARNING: touch_threshold_score — SIR confidence 0.68
        self.touch_threshold_score = touch_threshold_score
        # WARNING: tau — SIR confidence 0.55
        self.tau = tau

    def score(
        self,
        cohort_wallets: FrozenSet[str],
        intra_index: pd.DataFrame,
    ) -> Dict:
        """
        Compute EQ1 score for a single cohort.

        EQ1 (Section 4.2):
            score(C) = 10 * n_launches
                     + 5 / mean_first_rank
                     + sqrt(total_sol)

        Args:
            cohort_wallets: FrozenSet of wallet address strings.
            intra_index: Per-launch first-buyer index from IntraLaunchExtractor.

        Returns:
            dict with keys: wallets, n_launches, mean_first_rank, total_sol, score,
                            per_mint_hits (list of dicts for each touched launch).
        """
        wallet_set = set(cohort_wallets)

        # Filter intra_index to rows where wallet is in this cohort
        cohort_rows = intra_index[intra_index["wallet"].isin(wallet_set)]

        if cohort_rows.empty:
            return self._empty_score(cohort_wallets)

        # Group by mint to compute per-launch statistics
        per_mint_hits = []
        total_sol = 0.0
        per_launch_min_ranks = []

        for mint, group in cohort_rows.groupby("mint"):
            n_cohort_wallets_in_launch = group["wallet"].nunique()
            # Apply touch threshold: count launch only if >= threshold wallets present
            if n_cohort_wallets_in_launch < self.touch_threshold_score:
                continue

            min_rank = group["rank"].min()              # best (lowest) rank in this launch
            sol_in_launch = group["sol_committed"].sum()

            per_launch_min_ranks.append(min_rank)
            total_sol += sol_in_launch
            per_mint_hits.append({
                "mint": mint,
                "min_rank": int(min_rank),
                "block_time": int(group["block_time"].min()),
                "sol_committed": float(sol_in_launch),
            })

        n_launches = len(per_launch_min_ranks)
        if n_launches == 0:
            return self._empty_score(cohort_wallets)

        # EQ1 — mean_first_rank: mean of per-launch minimum ranks
        # ASSUMED: mean of min ranks per launch (SIR confidence 0.72)
        mean_first_rank = sum(per_launch_min_ranks) / n_launches

        # EQ1 computation
        # Term 1: 10 × number of launches touched
        term_coverage = 10.0 * n_launches
        # Term 2: 5 / mean_first_rank  (higher score for lower rank = better position)
        term_rank = 5.0 / mean_first_rank
        # Term 3: √(total SOL committed) — dampens outlier whales
        term_vol = math.sqrt(total_sol)

        score_val = term_coverage + term_rank + term_vol

        return {
            "wallets": sorted(cohort_wallets),
            "size": len(cohort_wallets),
            "n_launches_hit": n_launches,
            "mean_first_rank": round(mean_first_rank, 4),
            "total_sol": round(total_sol, 4),
            "score": round(score_val, 4),
            "per_mint_hits": per_mint_hits,
        }

    def score_all(
        self,
        components: List[FrozenSet[str]],
        intra_index: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score all candidate cohorts and apply the tau threshold.

        Args:
            components: List of frozensets from CohortSurface.surface().
            intra_index: Per-launch first-buyer index.

        Returns:
            pd.DataFrame of scored cohorts sorted by score DESC,
            with cohort_id assigned as COH-NNNN.
        """
        rows = [self.score(c, intra_index) for c in components]
        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # Apply score threshold tau
        df = self.apply_score_threshold(df)

        # Assign IDs sorted by descending score
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "cohort_id", [f"COH-{i+1:04d}" for i in range(len(df))])

        return df

    def apply_score_threshold(self, cohorts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Discard rows with score < tau.

        # WARNING: tau value undisclosed (SIR confidence 0.55).
        # Use calibrate() to binary-search for a target cohort count.
        """
        return cohorts_df[cohorts_df["score"] >= self.tau].copy()

    def calibrate(
        self,
        components: List[FrozenSet[str]],
        intra_index: pd.DataFrame,
        target_count: int = 1012,
        tol: int = 5,
    ) -> float:
        """
        Binary-search for the tau value that yields approximately *target_count* cohorts.

        This method exists to compensate for the undisclosed tau (SIR confidence 0.55).
        Writes the calibrated tau back to self.tau.

        Args:
            components: Candidate cohorts from CohortSurface.
            intra_index: Per-launch first-buyer index.
            target_count: Desired number of output cohorts (paper: 1,012).
            tol: Acceptable deviation from target_count.

        Returns:
            Calibrated tau value.
        """
        # Score everything once
        rows = [self.score(c, intra_index) for c in components]
        scores = sorted([r["score"] for r in rows if r["n_launches_hit"] > 0], reverse=True)

        if len(scores) <= target_count:
            self.tau = 0.0
            return 0.0

        # Binary search: find tau such that sum(s >= tau) ≈ target_count
        lo, hi = 0.0, max(scores)
        for _ in range(50):
            mid = (lo + hi) / 2
            count = sum(1 for s in scores if s >= mid)
            if abs(count - target_count) <= tol:
                self.tau = mid
                return mid
            elif count > target_count:
                lo = mid
            else:
                hi = mid

        self.tau = (lo + hi) / 2
        return self.tau

    @staticmethod
    def _empty_score(cohort_wallets: FrozenSet[str]) -> Dict:
        return {
            "wallets": sorted(cohort_wallets),
            "size": len(cohort_wallets),
            "n_launches_hit": 0,
            "mean_first_rank": float("inf"),
            "total_sol": 0.0,
            "score": 0.0,
            "per_mint_hits": [],
        }

    def __repr__(self) -> str:
        return f"CohortScorer(touch_threshold_score={self.touch_threshold_score}, tau={self.tau})"


class TierClassifier:
    """
    Assigns Standard / High / Premium tier labels based on n_launches_hit and score.

    Paper reference: Section 5 / Figure 3.
        Premium: n_launches >= 20   (22 cohorts in paper)
        High:    n_launches >= 10  OR  score >= 100  (153 cohorts including Premium)
        Standard: otherwise

    Args:
        premium_min_launches: Minimum launches for Premium tier (default: 20).
        high_min_launches: Minimum launches for High tier (default: 10).
        high_min_score: Minimum score for High tier (default: 100.0).
    """

    def __init__(
        self,
        premium_min_launches: int = 20,
        high_min_launches: int = 10,
        high_min_score: float = 100.0,
    ) -> None:
        self.premium_min_launches = premium_min_launches
        self.high_min_launches = high_min_launches
        self.high_min_score = high_min_score

    def classify_tier(self, row: pd.Series) -> str:
        """Classify a single cohort row into its tier."""
        n = row["n_launches_hit"]
        s = row["score"]
        if n >= self.premium_min_launches:
            return "premium"
        if n >= self.high_min_launches or s >= self.high_min_score:
            return "high"
        return "standard"

    def classify_all(self, cohorts_df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'tier' column to cohorts_df in place."""
        df = cohorts_df.copy()
        df["tier"] = df.apply(self.classify_tier, axis=1)
        return df

    def __repr__(self) -> str:
        return (
            f"TierClassifier(premium_min_launches={self.premium_min_launches}, "
            f"high_min_launches={self.high_min_launches}, "
            f"high_min_score={self.high_min_score})"
        )
