#!/usr/bin/env python3
"""
causal.py — Causal buyer-flow analysis entrypoint.

Runs the full causal analysis:
  - 3:1 random-matched design (Section 6.3)
  - Activity-matched placebo check (Appendix B.1 Design 2)
  - Uniform-random placebo check (Appendix B.1 Design 1)
  - Robustness checks (Appendix B.2-B.3)

Usage:
    python causal.py \\
        --buyers data/pumpfun_buyers.jsonl \\
        --launches data/pumpfun_launches.jsonl \\
        --cohorts results/sniper_cohorts.jsonl

Paper: Kamat (2026), Section 6 and Appendix B.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from red_cohort.causal.estimator import LiftEstimator
from red_cohort.causal.placebo import ActivityMatchedPlacebo, UniformRandomPlacebo
from red_cohort.causal.robustness import RobustnessChecker
from red_cohort.causal.sample import CausalSampleBuilder
from red_cohort.io.loader import DataLoader
from red_cohort.utils.config import PipelineConfig, set_seed
from red_cohort.utils.io_helpers import JsonlStreamer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RED-COHORT-2026: Causal buyer-flow analysis.")
    p.add_argument("--buyers", type=str, required=True)
    p.add_argument("--launches", type=str, required=True)
    p.add_argument("--cohorts", type=str, required=True,
                   help="Path to sniper_cohorts.jsonl from detect.py")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--skip-placebo", action="store_true",
                   help="Skip placebo checks (faster, for debugging)")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.causal.random_seed = args.seed
    set_seed(cfg.causal.random_seed)

    out = Path(cfg.data.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(chunk_size=cfg.hardware.chunk_size)

    print("[causal] Loading data...")
    buyers_df = loader.load_buyers(args.buyers)
    launches_df = loader.load_launches(args.launches)
    cohorts_records = list(JsonlStreamer.stream(args.cohorts))
    cohorts_df = pd.DataFrame(cohorts_records)
    print(f"  {len(buyers_df):,} buyer events | {len(launches_df):,} launches | {len(cohorts_df):,} cohorts")

    # Build intra index for sample construction
    from red_cohort.detection.extractor import IntraLaunchExtractor
    extractor = IntraLaunchExtractor(window_size=cfg.detection.first_buyer_window)
    print("[causal] Building intra-launch index...")
    intra_index = extractor.extract(buyers_df)

    # Sample construction
    print("[causal] Building treated / control samples...")
    sampler = CausalSampleBuilder(cfg.causal)
    treated_df, control_df = sampler.build(cohorts_df, intra_index, buyers_df, launches_df)
    print(f"  Treated: {len(treated_df):,} | Control: {len(control_df):,}")

    # Lift estimation
    estimator = LiftEstimator(cfg.causal)
    outcome_cols = ["first_30min_buyer_count", "first_30min_sol_inflow"]
    results = []

    for col in outcome_cols:
        print(f"[causal] Estimating lift: {col} ...")
        res = estimator.estimate(treated_df, control_df, col)
        res["design"] = "real_cohort_random_matched"
        results.append(res)
        print(f"  Lift: +{res['point_estimate_pct']:.1f}% "
              f"[{res['ci_lower_pct']:.1f}%, {res['ci_upper_pct']:.1f}%]")

    # Placebo checks
    if not args.skip_placebo:
        # Design 1: Uniform random
        print("[causal] Design 1 placebo (uniform random)...")
        uni_placebo = UniformRandomPlacebo(cfg.causal)
        uni_df = uni_placebo.build(cohorts_df, buyers_df)
        print(f"  Built {len(uni_df):,} uniform placebo cohorts.")

        # Design 2: Activity matched
        print("[causal] Design 2 placebo (activity matched) — preferred null...")
        act_placebo = ActivityMatchedPlacebo(cfg.causal)
        act_df = act_placebo.build(cohorts_df, buyers_df)
        print(f"  Built {len(act_df):,} activity-matched placebo cohorts.")

        # Build placebo treated sample using same >=2 wallet threshold
        act_treated_mints = sampler.build_treated(act_df, intra_index, touch_threshold=2)
        act_control_mints = sampler.build_control(
            intra_index["mint"].unique().tolist(), act_treated_mints,
            ratio=cfg.causal.control_ratio, seed=cfg.causal.random_seed
        )
        act_treated_outcomes = sampler.attach_outcomes(act_treated_mints, buyers_df, launches_df)
        act_control_outcomes = sampler.attach_outcomes(act_control_mints, buyers_df, launches_df)

        for col in outcome_cols:
            if len(act_treated_outcomes) >= 5:
                res = estimator.estimate(act_treated_outcomes, act_control_outcomes, col)
                res["design"] = "activity_matched_placebo"
                results.append(res)
                print(f"  [Placebo D2] {col}: +{res['point_estimate_pct']:.1f}% "
                      f"[{res['ci_lower_pct']:.1f}%, {res['ci_upper_pct']:.1f}%]")

    # Robustness checks
    print("[causal] Running robustness checks...")
    checker = RobustnessChecker(cfg.causal)
    cohorts_excl = checker.top_k_exclusion(cohorts_df, k=cfg.causal.top_k_exclusion)
    treated_excl, control_excl = sampler.build(cohorts_excl, intra_index, buyers_df, launches_df)
    for col in outcome_cols:
        if len(treated_excl) >= 5:
            res = estimator.estimate(treated_excl, control_excl, col)
            res["design"] = f"top{cfg.causal.top_k_exclusion}_excluded"
            results.append(res)

    # Save outputs
    results_df = pd.DataFrame(results)
    output_path = args.output or str(out / "causal_buyer_flow.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n[causal] Results → {output_path}")

    # Text summary (Appendix B.1 format)
    summary_path = str(out / "causal_buyer_flow_summary.txt")
    with open(summary_path, "w") as f:
        f.write("RED-COHORT-2026 Causal Buyer-Flow Results\n")
        f.write("=" * 50 + "\n\n")
        for _, row in results_df.iterrows():
            f.write(f"Design: {row['design']}\n")
            f.write(f"  Outcome: {row['outcome_col']}\n")
            f.write(f"  Treated mean: {row['treated_mean']:.2f}\n")
            f.write(f"  Control mean: {row['control_mean']:.2f}\n")
            f.write(f"  Lift: +{row['point_estimate_pct']:.1f}% "
                    f"[{row['ci_lower_pct']:.1f}%, {row['ci_upper_pct']:.1f}%]\n")
            f.write(f"  N treated: {row['n_treated']}, N control: {row['n_control']}\n\n")
    print(f"[causal] Summary → {summary_path}")


if __name__ == "__main__":
    main()
