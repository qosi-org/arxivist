#!/usr/bin/env python3
"""
run_evaluation.py
Compute all detection metrics from saved run_detection.py output.
Reproduces Tables 2 and 3 of arXiv:2605.11645.

Usage:
    python run_evaluation.py --results_dir results/detection/ --output results/eval_table.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from geomherd.evaluation.metrics import DetectionMetrics
from geomherd.utils.config import set_global_seed


def load_results(results_dir: str) -> List[dict]:
    all_results = []
    for p in Path(results_dir).glob("*.json"):
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_results.extend(data)
    return all_results


def compute_table(results: List[dict]) -> dict:
    """Compute Table 3 metrics: precision, recall_super, FAR_sub, AUROC, median lead."""
    alarm_plus = [r.get("alarm_plus_t") for r in results]
    alarm_minus = [r.get("alarm_minus_t") for r in results]
    event_times = [r.get("herding_event_t") for r in results]
    is_super = [r.get("is_supercritical", False) for r in results]

    # GeomHerd kappa_bar_plus
    pr_plus = DetectionMetrics.precision_recall_far(alarm_plus, event_times, is_super)
    lead_plus = DetectionMetrics.conditional_lead_time(alarm_plus, event_times)

    # GeomHerd beta_minus
    pr_minus = DetectionMetrics.precision_recall_far(alarm_minus, event_times, is_super)
    lead_minus = DetectionMetrics.conditional_lead_time(alarm_minus, event_times)

    return {
        "n_trajectories": len(results),
        "n_supercritical": sum(is_super),
        "n_subcritical": sum(1 for x in is_super if not x),
        "geomherd_kappa_plus": {**pr_plus, **lead_plus},
        "geomherd_beta_minus": {**pr_minus, **lead_minus},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/detection/")
    parser.add_argument("--output", type=str, default="results/eval_table.json")
    parser.add_argument("--n_boot", type=int, default=5000)
    args = parser.parse_args()

    set_global_seed(0)
    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} trajectories.")
    table = compute_table(results)

    print("\n=== Detection Results ===")
    print(f"{'Detector':<25} {'Prec':>6} {'Rec':>6} {'FAR':>6} {'MedLead':>8} {'CI95':>18} {'N':>4}")
    print("-" * 75)
    for key, name in [("geomherd_kappa_plus", "GeomHerd κ̄⁺_OR"),
                       ("geomherd_beta_minus", "GeomHerd β⁻")]:
        m = table[key]
        ci = f"[{m.get('ci_lower', float('nan')):.0f}, {m.get('ci_upper', float('nan')):.0f}]"
        print(f"{name:<25} {m.get('precision', 0):.2f}  "
              f"{m.get('recall_super', 0):.2f}  "
              f"{m.get('far_sub', 0):.2f}  "
              f"{m.get('median_lead', float('nan')):8.0f}  "
              f"{ci:>18}  {m.get('n_paired', 0):4d}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(table, f, indent=2, default=str)
    print(f"\nFull table saved to {args.output}")


if __name__ == "__main__":
    main()
