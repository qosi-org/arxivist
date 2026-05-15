"""
FutureSim — Offline Scoring Script
=====================================
Compute metrics from saved predictions and ground truth.

Usage:
  python score_results.py --predictions results/predictions.json \
                           --ground-truth data/ground_truth.json \
                           --output results/metrics.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from futuresim.scoring.brier import (
    compute_brier_skill_score,
    compute_accuracy,
    compute_time_weighted_score,
)
from futuresim.utils.logging import get_logger

logger = get_logger("futuresim.scorer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score FutureSim predictions offline")
    p.add_argument("--predictions", required=True, help="Path to predictions JSON/CSV")
    p.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    p.add_argument("--output", default="metrics.json", help="Output metrics path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)  # {qid: {outcome: probability}}

    with open(args.ground_truth) as f:
        ground_truths = json.load(f)  # {qid: answer_string}

    bss_scores = []
    accuracy_scores = []

    for qid, gt in ground_truths.items():
        pred = predictions.get(str(qid), {})
        bss = compute_brier_skill_score(pred, gt)
        acc = compute_accuracy(pred, gt)
        bss_scores.append(bss)
        accuracy_scores.append(acc)

    n = len(bss_scores)
    metrics = {
        "num_questions": n,
        "mean_brier_skill_score": sum(bss_scores) / n if n else 0.0,
        "accuracy": sum(accuracy_scores) / n if n else 0.0,
        "positive_bss_fraction": sum(1 for s in bss_scores if s > 0) / n if n else 0.0,
    }

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to {args.output}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
