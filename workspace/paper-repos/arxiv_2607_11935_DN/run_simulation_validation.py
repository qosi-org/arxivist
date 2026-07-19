#!/usr/bin/env python3
"""
run_simulation_validation.py -- Run all six simulated tipping-point systems
and reproduce Table 3 and Figure 3 of arXiv:2607.11935.

Example:
    python run_simulation_validation.py --config configs/config.yaml --system all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ews_kalman.evaluation import SimulationValidator
from ews_kalman.utils import load_config, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--system", type=str, default="all",
        choices=[
            "fold_bifurcation", "beta_step_change", "beta_linear_decay",
            "logistic_map", "stommel_amoc", "critical_slowing_down", "all",
        ],
        help="Which simulated system to run",
    )
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to write tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.hardware["seed"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_config = cfg.evaluation.get("simulation", {})
    validator = SimulationValidator()

    print("Running six simulated tipping-point systems (Table 3 reproduction)...")
    print(
        "NOTE: noise magnitudes, forcing schedules, and simulation lengths are "
        "not given numerically in the paper (Section 2.4) -- illustrative "
        "parameters from configs/config.yaml are used. See README.md "
        "'Reproducibility Notes' before comparing exact lead-time values.\n"
    )

    results = validator.validate_all_systems(sim_config=sim_config, seed=cfg.hardware["seed"])

    if args.system != "all":
        name_map = {
            "fold_bifurcation": "Fold bifurcation",
            "beta_step_change": "Beta step change",
            "beta_linear_decay": "Beta linear decay",
            "logistic_map": "Logistic map",
            "stommel_amoc": "Stommel AMOC",
            "critical_slowing_down": "Critical slowing down",
        }
        results = [r for r in results if r["simulation"] == name_map[args.system]]

    print(f"{'Simulation':<22} {'Tipping t':>10} {'beta lead':>10} {'AR1 lead':>10} {'Winner':>10}")
    print("-" * 66)
    for r in results:
        beta_lead_str = str(r["beta_lead"]) if r["beta_lead"] is not None else "-"
        ar1_lead_str = str(r["ar1_lead"]) if r["ar1_lead"] is not None else "-"
        print(
            f"{r['simulation']:<22} {r['tipping_t']:>10} {beta_lead_str:>10} "
            f"{ar1_lead_str:>10} {r['winner']:>10}"
        )

    with open(output_dir / "table3_simulation_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTable written to {output_dir / 'table3_simulation_validation.json'}")


if __name__ == "__main__":
    main()
