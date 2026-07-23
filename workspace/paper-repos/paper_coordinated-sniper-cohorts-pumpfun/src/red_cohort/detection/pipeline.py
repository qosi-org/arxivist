"""
detection/pipeline.py
----------------------
Orchestrates the full Stage 1 → Stage 2 detection pipeline as a single callable.

Paper: Kamat (2026), Sections 4.1–4.2.
Pipeline:
    buyers.jsonl → Stage1_Extract → Graph → Filter → UnionFind → SizeFilter → Score → Tier
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from red_cohort.detection.extractor import IntraLaunchExtractor
from red_cohort.detection.graph import CoOccurrenceGraph
from red_cohort.detection.scorer import CohortScorer, TierClassifier
from red_cohort.detection.union_find import CohortSurface
from red_cohort.utils.config import DetectionConfig, TierConfig


class DetectionPipeline:
    """
    End-to-end detection pipeline from raw buyer events to scored cohort catalogue.

    Args:
        detection_cfg: Detection hyperparameters (edge cutoff, max size, tau, etc.).
        tier_cfg: Tier classification thresholds.
        verbose: Print progress messages if True.

    Paper reference: Sections 4.1–4.2, Figure 1.

    Expected output: ~1,012 cohorts at default settings.
    """

    def __init__(
        self,
        detection_cfg: DetectionConfig,
        tier_cfg: TierConfig,
        verbose: bool = True,
    ) -> None:
        self.cfg = detection_cfg
        self.tier_cfg = tier_cfg
        self.verbose = verbose

        self.extractor = IntraLaunchExtractor(window_size=detection_cfg.first_buyer_window)
        self.graph_builder = CoOccurrenceGraph(min_weight=detection_cfg.edge_weight_cutoff)
        self.surfer = CohortSurface(max_size=detection_cfg.max_cohort_size)
        self.scorer = CohortScorer(
            touch_threshold_score=detection_cfg.touch_threshold_score,
            tau=detection_cfg.score_tau,
        )
        self.tier_classifier = TierClassifier(
            premium_min_launches=tier_cfg.premium_min_launches,
            high_min_launches=tier_cfg.high_min_launches,
            high_min_score=tier_cfg.high_min_score,
        )

    def run(
        self,
        buyers_df: pd.DataFrame,
        intra_index: Optional[pd.DataFrame] = None,
        calibrate: bool = False,
        calibrate_target: int = 1012,
    ) -> pd.DataFrame:
        """
        Run the full detection pipeline.

        Args:
            buyers_df: Raw buyer events from DataLoader.load_buyers().
            intra_index: Pre-computed Stage-1 index (bypass Stage 1 if provided).
            calibrate: If True, binary-search for tau to match calibrate_target cohorts.
            calibrate_target: Target cohort count for calibration (paper: 1,012).

        Returns:
            cohorts_df: DataFrame of scored, tiered cohorts.
        """
        # Stage 1: Extract first-buyer window
        if intra_index is None:
            self._log("Stage 1: Extracting first-buyer window...")
            intra_index = self.extractor.extract(buyers_df)
        else:
            self._log("Stage 1: Using pre-computed intra_index (bypassed).")

        n_qualifying = intra_index["mint"].nunique()
        self._log(f"  Qualifying mints: {n_qualifying:,}")

        # Stage 2a: Build co-occurrence graph
        self._log(f"Stage 2a: Building co-occurrence graph (cutoff={self.cfg.edge_weight_cutoff})...")
        G_filtered = self.graph_builder.build(intra_index)
        self._log(f"  Edges after filter: {G_filtered.number_of_edges():,} | Nodes: {G_filtered.number_of_nodes():,}")

        # Stage 2b: Union-find + size filter
        self._log("Stage 2b: Union-find + size filter...")
        components = self.surfer.surface(G_filtered)
        self._log(f"  Cohort candidates after size filter: {len(components):,}")

        # Stage 2c: Calibrate tau if requested
        if calibrate:
            self._log(f"Calibrating tau for target={calibrate_target} cohorts...")
            tau = self.scorer.calibrate(components, intra_index, target_count=calibrate_target)
            self._log(f"  Calibrated tau = {tau:.4f}")

        # Stage 2d: Score and apply tau threshold
        self._log(f"Stage 2c: Scoring cohorts (tau={self.scorer.tau})...")
        cohorts_df = self.scorer.score_all(components, intra_index)
        self._log(f"  Cohorts after tau filter: {len(cohorts_df):,}")

        # Stage 2e: Assign tiers
        cohorts_df = self.tier_classifier.classify_all(cohorts_df)

        # Summary
        if not cohorts_df.empty:
            n_premium = (cohorts_df["tier"] == "premium").sum()
            n_high = (cohorts_df["tier"] == "high").sum()
            n_standard = (cohorts_df["tier"] == "standard").sum()
            self._log(
                f"Detection complete: {len(cohorts_df):,} cohorts "
                f"(premium={n_premium}, high={n_high}, standard={n_standard})"
            )

        return cohorts_df

    def run_ablation(
        self,
        buyers_df: pd.DataFrame,
        cutoffs: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Run detection at multiple edge-weight cutoffs for Appendix A ablation.

        Paper: Appendix A — cutoffs [2, 3, 5] → [1,562, 1,161, 737] raw components.

        Args:
            buyers_df: Raw buyer events.
            cutoffs: List of edge-weight cutoffs to evaluate. Default: [2, 3, 5].

        Returns:
            Dict mapping cutoff → cohorts_df.
        """
        if cutoffs is None:
            cutoffs = self.cfg.ablation_cutoffs

        self._log("Stage 1: Extracting first-buyer window (shared across ablations)...")
        intra_index = self.extractor.extract(buyers_df)

        # Build raw graph once with min_weight=1 (no filter yet)
        raw_graph_builder = CoOccurrenceGraph(min_weight=1)
        G_raw = raw_graph_builder.build(intra_index)

        results = {}
        for cutoff in cutoffs:
            self._log(f"Ablation: cutoff={cutoff}...")
            G_filtered = self.graph_builder.filter_edges(G_raw, min_weight=cutoff)
            components = self.surfer.surface(G_filtered)
            scored = self.scorer.score_all(components, intra_index)
            tiered = self.tier_classifier.classify_all(scored)
            results[cutoff] = tiered
            self._log(f"  cutoff={cutoff}: {len(tiered):,} cohorts, {G_filtered.number_of_edges():,} qualifying pairs")

        return results

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DetectionPipeline] {msg}")

    def __repr__(self) -> str:
        return f"DetectionPipeline(cfg={self.cfg})"
