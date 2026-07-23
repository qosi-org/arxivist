#!/usr/bin/env python3
"""
run_readiness_survey.py -- Load and summarise the top-20 cryptocurrency
quantum-readiness ratings, reproducing Table 3 and Figure 5
(arXiv:2606.14484, Section 6).

Example:
    python run_readiness_survey.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_horizon.survey import QuantumReadinessSurvey
from quantum_horizon.utils import load_config, plot_figure5_readiness, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to write outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.hardware["seed"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    survey = QuantumReadinessSurvey()
    ratings_df = survey.load_ratings(cfg.readiness_survey["table3_path"])
    stats = survey.summary_stats(ratings_df)

    print(f"Loaded {len(ratings_df)} coin ratings\n")
    print(ratings_df[["coin", "rating"]].sort_values("rating", ascending=False).to_string(index=False))

    print(f"\nMean rating: {stats['mean_rating']:.2f}")
    print(f"Median rating: {stats['median_rating']:.2f}")
    print(f"Max rating: {stats['max_rating']:.1f}  (paper: 'none reaches 5')")
    print(f"No coin reaches 5: {stats['no_coin_reaches_5']}")

    ratings_df.to_csv(output_dir / "readiness_ratings.csv", index=False)

    summary_out = {k: v for k, v in stats.items() if k != "rating_counts"}
    summary_out["rating_counts"] = {str(k): v for k, v in stats["rating_counts"].items()}
    with open(output_dir / "readiness_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    plot_figure5_readiness(ratings_df, save_path=str(output_dir / "figure5_readiness.png"))
    print(f"\nResults written to {output_dir}/")


if __name__ == "__main__":
    main()
