"""
causal/placebo.py
------------------
Implements the two placebo designs from Appendix B.1.

Design 1 (UniformRandomPlacebo):
    Sample wallets uniformly from buyer-event universe.
    Paper: +152.0% lift — simpler but biased (high-activity wallets over-represented).

Design 2 (ActivityMatchedPlacebo):  ← preferred clean null
    For each real cohort of size k with per-wallet launch counts {a1..ak},
    sample k non-cohort wallets each matched to ai (within ±tolerance launches).
    Paper: +216.3% [+183.8%, +255.2%] — controls for per-wallet activity frequency.

Paper: Kamat (2026), Section 6.4 and Appendix B.1.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Tuple

import pandas as pd

from red_cohort.utils.config import CausalConfig


class UniformRandomPlacebo:
    """
    Design 1 placebo: sample 1,012 placeholder cohorts uniformly at random
    from the buyer-event universe, at the same size distribution as real cohorts.

    Note: This design does NOT control for per-wallet activity level.
    High-activity wallets are over-represented, inflating the placebo lift.
    Use ActivityMatchedPlacebo (Design 2) for the cleaner null.

    Paper reference: Appendix B.1 — Design 1. Lift: +152.0%.
    """

    def __init__(self, config: CausalConfig) -> None:
        self.cfg = config

    def build(
        self,
        cohorts_df: pd.DataFrame,
        buyers_df: pd.DataFrame,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build 1,012 uniform-random placebo cohorts at the same size distribution.

        Args:
            cohorts_df: Real cohort catalogue (for size distribution).
            buyers_df: Full buyer events (wallet pool).
            seed: Random seed (default from config).

        Returns:
            placebo_df: DataFrame with same schema as cohorts_df,
                        with randomly assigned wallet lists.
        """
        seed = seed if seed is not None else self.cfg.random_seed
        rng = random.Random(seed)

        all_wallets = buyers_df["wallet"].unique().tolist()
        real_sizes = cohorts_df["size"].tolist()

        rows = []
        for i, sz in enumerate(real_sizes):
            sampled = rng.sample(all_wallets, min(sz, len(all_wallets)))
            rows.append({
                "cohort_id": f"PLACEBO-UNI-{i+1:04d}",
                "wallets": sampled,
                "size": len(sampled),
            })

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return "UniformRandomPlacebo()"


class ActivityMatchedPlacebo:
    """
    Design 2 placebo (preferred): for each real cohort of size k with per-wallet
    launch counts {a1..ak}, sample k non-cohort wallets each individually matched
    to ai within ±tolerance launches.

    This controls for the wallet-activity-frequency confound, isolating
    coordination per se from generic frequent-trader behaviour.

    Paper reference:
        Appendix B.1 — Design 2. Lift: +216.3% [+183.8%, +255.2%].
        "The real-cohort point estimate falls well below the placebo
        confidence-interval lower bound, refuting a strong cohort-specific
        causal interpretation."

    Args:
        config: CausalConfig with activity_match_tolerance and random_seed.
    """

    def __init__(self, config: CausalConfig) -> None:
        self.cfg = config

    def _compute_wallet_launch_counts(
        self,
        buyers_df: pd.DataFrame,
    ) -> Dict[str, int]:
        """Compute number of distinct launches each wallet participated in."""
        return buyers_df.groupby("wallet")["mint"].nunique().to_dict()

    def match_wallet(
        self,
        target_count: int,
        candidate_pool: List[Tuple[str, int]],
        tolerance: int,
        rng: random.Random,
        used: set,
    ) -> Optional[str]:
        """
        Sample one wallet from candidate_pool whose launch count is within
        ±tolerance of target_count, excluding already-used wallets.

        Args:
            target_count: Target per-wallet launch count to match.
            candidate_pool: List of (wallet, launch_count) tuples (non-cohort wallets).
            tolerance: Matching window: |count - target_count| <= tolerance.
            rng: Seeded random.Random instance.
            used: Set of wallet addresses already assigned (prevents reuse).

        Returns:
            Matched wallet address, or None if no match found.
        """
        eligible = [
            w for w, c in candidate_pool
            if abs(c - target_count) <= tolerance and w not in used
        ]
        if not eligible:
            return None
        return rng.choice(eligible)

    def build(
        self,
        cohorts_df: pd.DataFrame,
        buyers_df: pd.DataFrame,
        match_tolerance: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build 1,012 activity-matched placebo cohorts.

        For each real cohort C of size k with wallets w1..wk:
          1. Get per-wallet launch count for each wi.
          2. Sample one non-cohort wallet for each wi, matched within ±tolerance.
          3. Form the placebo cohort from sampled wallets.

        Paper: "1,012 (all real cohorts matched successfully)" — Appendix B.1.

        Args:
            cohorts_df: Real cohort catalogue.
            buyers_df: Full buyer events (for non-cohort wallet pool).
            match_tolerance: Launch-count matching window (default from config: ±100).
            seed: Random seed (default from config: 42).

        Returns:
            placebo_df: DataFrame of matched placebo cohorts.
        """
        tolerance = match_tolerance if match_tolerance is not None else self.cfg.activity_match_tolerance
        seed = seed if seed is not None else self.cfg.random_seed
        rng = random.Random(seed)

        # Build non-cohort wallet pool
        all_cohort_wallets: set = set()
        for wallets in cohorts_df["wallets"]:
            all_cohort_wallets.update(wallets)

        wallet_counts = self._compute_wallet_launch_counts(buyers_df)
        non_cohort_pool: List[Tuple[str, int]] = [
            (w, c) for w, c in wallet_counts.items()
            if w not in all_cohort_wallets
        ]

        # Real cohort per-wallet launch counts
        real_wallet_counts = {w: wallet_counts.get(w, 0) for w in all_cohort_wallets}

        rows = []
        used_wallets: set = set()

        for idx, row in cohorts_df.iterrows():
            real_wallets = row["wallets"]
            placebo_wallets = []
            success = True

            for rw in real_wallets:
                target = real_wallet_counts.get(rw, 0)
                matched = self.match_wallet(target, non_cohort_pool, tolerance, rng, used_wallets)
                if matched is None:
                    success = False
                    break
                placebo_wallets.append(matched)
                used_wallets.add(matched)

            if success and len(placebo_wallets) >= 2:
                rows.append({
                    "cohort_id": f"PLACEBO-ACT-{idx+1:04d}",
                    "wallets": placebo_wallets,
                    "size": len(placebo_wallets),
                    "matched_from": row["cohort_id"],
                })

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"ActivityMatchedPlacebo(tolerance={self.cfg.activity_match_tolerance})"
