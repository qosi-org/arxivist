#!/usr/bin/env python3
"""
analyze.py — Descriptive statistics and figures entrypoint.

Loads the cohort catalogue and produces Tables 1-3 plus Figures 1-3.

Usage:
    python analyze.py --cohorts results/sniper_cohorts.jsonl
    python analyze.py --cohorts results/sniper_cohorts.jsonl --top-k 20

Paper: Kamat (2026), Section 5.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from red_cohort.analysis.descriptive import DescriptiveAnalyzer
from red_cohort.analysis.visualizer import Visualizer
from red_cohort.utils.io_helpers import JsonlStreamer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce descriptive stats and figures.")
    p.add_argument("--cohorts", type=str, required=True,
                   help="Path to sniper_cohorts.jsonl")
    p.add_argument("--output-dir", type=str, default="results/",
                   help="Output directory for figures and tables")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of top cohorts in Table 1 (default: 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] Loading cohorts from {args.cohorts}")
    records = list(JsonlStreamer.stream(args.cohorts))
    cohorts_df = pd.DataFrame(records)

    analyzer = DescriptiveAnalyzer()
    viz = Visualizer()

    # Table 1
    top_k = analyzer.top_k_cohorts(cohorts_df, k=args.top_k)
    t1_path = str(out / "table1_top10_cohorts.csv")
    top_k.to_csv(t1_path, index=False)
    print(f"[analyze] Table 1 (top-{args.top_k}) → {t1_path}")
    print(top_k.to_string(index=False))

    # Table 2
    size_dist = analyzer.size_distribution(cohorts_df)
    t2_path = str(out / "table2_size_distribution.csv")
    size_dist.to_csv(t2_path, index=False)
    print(f"\n[analyze] Table 2 (size distribution) → {t2_path}")

    # Table 3
    stats = analyzer.headline_stats(cohorts_df)
    t3_path = str(out / "table3_descriptive_stats.csv")
    pd.DataFrame([stats]).to_csv(t3_path, index=False)
    print(f"[analyze] Table 3 (headline stats) → {t3_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Figure 1
    viz.fig1_size_distribution(cohorts_df, str(out / "fig1_size_distribution.svg"))

    # Figure 2
    lorenz = analyzer.lorenz_data(cohorts_df)
    viz.fig2_lorenz_curve(lorenz, str(out / "fig2_lorenz_curve.svg"))

    # Figure 3
    viz.fig3_score_vs_launches(cohorts_df, str(out / "fig3_score_vs_launches.svg"))

    print(f"\n[analyze] Done. All outputs in {out}/")


if __name__ == "__main__":
    main()
