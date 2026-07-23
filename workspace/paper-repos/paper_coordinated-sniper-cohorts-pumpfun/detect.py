#!/usr/bin/env python3
"""
detect.py — Stage 1+2 detection pipeline entrypoint.

Runs the full two-stage detection pipeline from raw buyer events to
a scored, tiered cohort catalogue (sniper_cohorts.jsonl).

Usage:
    python detect.py --buyers data/pumpfun_buyers.jsonl
    python detect.py --buyers data/pumpfun_buyers.jsonl --ablation
    python detect.py --from-intra data/sniper_cohorts_intra.jsonl.gz --calibrate

Paper: Kamat (2026), Sections 4.1–4.2.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from red_cohort.detection.pipeline import DetectionPipeline
from red_cohort.io.loader import DataLoader
from red_cohort.utils.config import PipelineConfig, set_seed
from red_cohort.utils.io_helpers import JsonlStreamer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RED-COHORT-2026: Two-stage cohort detection pipeline."
    )
    p.add_argument("--buyers", type=str, help="Path to pumpfun_buyers.jsonl")
    p.add_argument("--from-intra", type=str, default=None,
                   help="Skip Stage 1 by loading pre-computed sniper_cohorts_intra.jsonl.gz")
    p.add_argument("--config", type=str, default="configs/config.yaml",
                   help="Config YAML path (default: configs/config.yaml)")
    p.add_argument("--output", type=str, default=None,
                   help="Output cohort catalogue JSONL path")
    p.add_argument("--ablation", action="store_true",
                   help="Run edge-weight ablation across cutoffs [2, 3, 5] (Appendix A)")
    p.add_argument("--cutoff", type=int, default=None,
                   help="Override edge_weight_cutoff for a single run")
    p.add_argument("--calibrate", action="store_true",
                   help="Binary-search tau to match target cohort count (default: 1,012)")
    p.add_argument("--calibrate-target", type=int, default=1012,
                   help="Target cohort count for --calibrate (default: 1012)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed override")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig.from_yaml(args.config)

    if args.seed is not None:
        cfg.causal.random_seed = args.seed
    if args.cutoff is not None:
        cfg.detection.edge_weight_cutoff = args.cutoff

    set_seed(cfg.causal.random_seed)

    output_path = args.output or str(Path(cfg.data.output_dir) / "sniper_cohorts.jsonl")
    Path(cfg.data.output_dir).mkdir(parents=True, exist_ok=True)

    loader = DataLoader(chunk_size=cfg.hardware.chunk_size)

    # Load data
    if args.from_intra:
        print(f"[detect] Loading pre-computed intra index: {args.from_intra}")
        intra_index = loader.load_intra(args.from_intra)
        buyers_df = None
    else:
        buyers_path = args.buyers or cfg.data.buyers_path
        print(f"[detect] Loading buyers: {buyers_path}")
        buyers_df = loader.load_buyers(buyers_path)
        intra_index = None

    pipeline = DetectionPipeline(
        detection_cfg=cfg.detection,
        tier_cfg=cfg.tier,
        verbose=not args.quiet,
    )

    if args.ablation:
        print("[detect] Running ablation across cutoffs:", cfg.detection.ablation_cutoffs)
        ablation_results = pipeline.run_ablation(
            buyers_df=buyers_df or loader.load_buyers(cfg.data.buyers_path),
            cutoffs=cfg.detection.ablation_cutoffs,
        )
        ablation_rows = []
        for cutoff, df in ablation_results.items():
            ablation_rows.append({
                "edge_weight_cutoff": cutoff,
                "n_cohorts": len(df),
                "n_qualifying_pairs": "see graph stats",
            })
        ablation_path = str(Path(cfg.data.output_dir) / "appendix_a_ablations.csv")
        import pandas as pd
        pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)
        print(f"[detect] Ablation results saved to {ablation_path}")
        return

    cohorts_df = pipeline.run(
        buyers_df=buyers_df,
        intra_index=intra_index,
        calibrate=args.calibrate,
        calibrate_target=args.calibrate_target,
    )

    # Serialize to JSONL
    records = cohorts_df.to_dict(orient="records")
    JsonlStreamer.write(records, output_path)
    print(f"[detect] Cohort catalogue written: {output_path} ({len(records):,} cohorts)")

    # Also write Table 2 and Table 3
    from red_cohort.analysis.descriptive import DescriptiveAnalyzer
    analyzer = DescriptiveAnalyzer()
    size_dist = analyzer.size_distribution(cohorts_df)
    stats = analyzer.headline_stats(cohorts_df)

    size_path = str(Path(cfg.data.output_dir) / "table2_size_distribution.csv")
    size_dist.to_csv(size_path, index=False)

    stats_path = str(Path(cfg.data.output_dir) / "table3_descriptive_stats.csv")
    import pandas as pd
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    print(f"[detect] Table 2 → {size_path}")
    print(f"[detect] Table 3 → {stats_path}")


if __name__ == "__main__":
    main()
