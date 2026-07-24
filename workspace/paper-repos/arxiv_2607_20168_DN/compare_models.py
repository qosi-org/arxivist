#!/usr/bin/env python3
"""
Pairwise significance and Holm family-wise correction across saved per-window
IC series (Sec 5.2), reproducing the paper's Table 3/4/5-style comparisons.

Example:
    python compare_models.py --results-dir results/ --output results/comparison_table.csv
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qkernel_finance.evaluation.metrics import PerformanceMetrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model IC series with paired significance + Holm correction.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory of *_ic_series.csv files from run_study.py")
    parser.add_argument("--output", type=str, default="results/comparison_table.csv", help="Where to write the comparison table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = PerformanceMetrics()

    series_files = sorted(Path(args.results_dir).glob("*_ic_series.csv"))
    if len(series_files) < 2:
        raise SystemExit(f"Need at least 2 *_ic_series.csv files in {args.results_dir} to compare (found {len(series_files)}). Run run_study.py first.")

    series = {f.stem.replace("_ic_series", ""): pd.read_csv(f)["ic"].values for f in series_files}
    lengths = {k: len(v) for k, v in series.items()}
    if len(set(lengths.values())) > 1:
        print(f"WARNING: model IC series have different lengths ({lengths}) -- pairwise comparisons only use the overlapping window count.")

    summary_rows = []
    for name, ic in series.items():
        s = metrics.summarize(ic)
        summary_rows.append({"model": name, "n_windows": len(ic), **s})
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_ic", ascending=False)
    print("\n=== Model summary (analogous to Table 3/5) ===")
    print(summary_df.to_string(index=False))

    pair_rows = []
    for name_a, name_b in combinations(series.keys(), 2):
        n = min(len(series[name_a]), len(series[name_b]))
        if n < 2:
            continue
        result = metrics.paired_significance(series[name_a][:n], series[name_b][:n])
        pair_rows.append({"model_a": name_a, "model_b": name_b, **result})

    if pair_rows:
        pair_df = pd.DataFrame(pair_rows)
        pair_df["holm_adjusted_t_p"] = metrics.holm_correct(pair_df["paired_t_p"].tolist())
        print("\n=== Pairwise comparisons (Sec 5.2: Holm family-wise correction) ===")
        print(pair_df.to_string(index=False))
    else:
        pair_df = pd.DataFrame()
        print("\nNo pairwise comparisons possible (need >=2 windows of overlap between at least 2 models).")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    if not pair_df.empty:
        pair_df.to_csv(str(args.output).replace(".csv", "_pairwise.csv"), index=False)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
