#!/usr/bin/env python3
"""
run_migration_race.py -- Run the migration-race (Mosca's inequality)
scenarios and reproduce Figure 4 (arXiv:2606.14484, Section 7.4).

Example:
    python run_migration_race.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_horizon.migration import MigrationRaceModel
from quantum_horizon.utils import load_config, plot_figure4_migration_race, set_global_seed


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

    mig_cfg = cfg.migration
    model = MigrationRaceModel()

    scenarios = {}
    for name, s in mig_cfg["scenarios"].items():
        if "reported_finish_year" in s:
            duration = s["reported_finish_year"] - s["start_year"]
        else:
            finish_mid = (s["reported_finish_year_min"] + s["reported_finish_year_max"]) / 2
            duration = finish_mid - s["start_year"]
        scenarios[name] = {"start_year": s["start_year"], "migration_duration_years": duration}

    df = model.run_scenarios(scenarios, mig_cfg["crqc_estimates"])
    print("Migration-race scenarios:\n")
    print(df.to_string(index=False))

    at_risk = df[df["at_risk"]]
    print(f"\nAt-risk combinations: {len(at_risk)} / {len(df)}")
    if len(at_risk) > 0:
        print(at_risk[["scenario", "crqc_estimate_name"]].to_string(index=False))
    print(
        "(Paper: 'the only at-risk case in the entire sweep is a severely "
        "delayed start... running against an early machine')"
    )

    df.to_csv(output_dir / "migration_race_scenarios.csv", index=False)
    with open(output_dir / "migration_race_summary.json", "w") as f:
        json.dump({"n_scenarios": len(df), "n_at_risk": len(at_risk)}, f, indent=2)

    plot_figure4_migration_race(
        df, mig_cfg["crqc_estimates"], save_path=str(output_dir / "figure4_migration_race.png")
    )
    print(f"\nResults written to {output_dir}/")


if __name__ == "__main__":
    main()
