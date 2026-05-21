"""
Aggregate Metrics and Reporting
=================================
Computes and formats simulation-level metrics for FutureSim.

Paper reference: Section 3 (Evaluation), Appendix C (Metrics)
Reports: mean BSS, accuracy, time-weighted score, and per-topic breakdowns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from futuresim.scoring.brier import (
    compute_brier_skill_score,
    compute_accuracy,
    compute_time_weighted_score,
)


def compute_simulation_metrics(
    market_df: pd.DataFrame,
    daily_predictions: Optional[dict] = None,
) -> dict:
    """
    Compute all simulation metrics over the resolved question set.

    Args:
        market_df: Final market CSV dataframe with is_resolved=True rows
        daily_predictions: Optional {qid: {day: {outcome: prob}}} for time-weighted score

    Returns:
        Dict with mean_bss, accuracy, positive_bss_fraction, per_topic_accuracy
    """
    resolved = market_df[market_df["is_resolved"]].copy()
    if resolved.empty:
        return {"num_resolved": 0, "mean_bss": 0.0, "accuracy": 0.0}

    bss_list, acc_list = [], []
    for _, row in resolved.iterrows():
        pred = json.loads(row["my_prediction"]) if pd.notna(row["my_prediction"]) else {}
        gt = str(row["ground_truth"]) if pd.notna(row["ground_truth"]) else ""
        bss_list.append(compute_brier_skill_score(pred, gt))
        acc_list.append(compute_accuracy(pred, gt))

    n = len(bss_list)
    metrics = {
        "num_resolved": n,
        "num_total": len(market_df),
        "mean_bss": round(sum(bss_list) / n, 6),
        "accuracy": round(sum(acc_list) / n, 6),
        "positive_bss_fraction": round(sum(1 for s in bss_list if s > 0) / n, 4),
        "negative_bss_fraction": round(sum(1 for s in bss_list if s < 0) / n, 4),
        "overconfidence_rate_50": round(
            # Fraction of wrong top predictions that were assigned ≥ 0.5 probability
            # Paper Section 4.3: "27.4% assign at least 0.5 probability to the wrong top answer"
            sum(
                1 for _, row in resolved.iterrows()
                if _is_overconfident(row, threshold=0.5)
            ) / max(sum(acc_list == 0 for acc_list in acc_list), 1),
            4,
        ),
    }

    # Per-topic breakdown (if topic column present)
    if "topic" in resolved.columns:
        topics = {}
        for topic, group in resolved.groupby("topic"):
            t_bss = []
            t_acc = []
            for _, row in group.iterrows():
                pred = json.loads(row["my_prediction"]) if pd.notna(row["my_prediction"]) else {}
                gt = str(row["ground_truth"]) if pd.notna(row["ground_truth"]) else ""
                t_bss.append(compute_brier_skill_score(pred, gt))
                t_acc.append(compute_accuracy(pred, gt))
            topics[str(topic)] = {
                "n": len(t_bss),
                "mean_bss": round(sum(t_bss) / len(t_bss), 4),
                "accuracy": round(sum(t_acc) / len(t_acc), 4),
            }
        metrics["per_topic"] = topics

    return metrics


def _is_overconfident(row, threshold: float = 0.5) -> bool:
    """Return True if the top predicted outcome is wrong AND has probability >= threshold."""
    pred = json.loads(row["my_prediction"]) if pd.notna(row.get("my_prediction")) else {}
    gt = str(row.get("ground_truth", ""))
    if not pred:
        return False
    top_outcome = max(pred, key=pred.get)
    top_prob = pred[top_outcome]
    is_correct = top_outcome.strip().lower() == gt.strip().lower()
    return (not is_correct) and (top_prob >= threshold)


def print_leaderboard(results: list[dict]) -> None:
    """
    Print a leaderboard table matching Figure 1 of the paper.

    Args:
        results: List of dicts with keys: model, harness, accuracy, bss
    """
    print("\n" + "=" * 60)
    print("FutureSim Leaderboard")
    print("=" * 60)
    print(f"{'Model':<22} {'Harness':<14} {'Accuracy':>10} {'BSS':>8}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: -x.get("accuracy", 0)):
        model = r.get("model", "?")
        harness = r.get("harness", "?")
        acc = r.get("accuracy", 0.0)
        bss = r.get("bss", 0.0)
        print(f"{model:<22} {harness:<14} {acc:>9.1%} {bss:>8.3f}")
    print("=" * 60 + "\n")


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save metrics dict to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")
