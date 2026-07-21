#!/usr/bin/env python3
"""
run_all.py -- Convenience wrapper running all five analyses in sequence,
reproducing Figures 1-5 and every headline number in arXiv:2606.14484.

Example:
    python run_all.py --config configs/config.yaml --output-dir results/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to write all outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).parent

    steps = [
        ("1/5: Timeline forecast (Figure 2)", "run_timeline_forecast.py", ["--sensitivity-sweep"]),
        ("2/5: Mining analysis (Figure 1)", "run_mining_analysis.py", []),
        ("3/5: Exposure analysis (Figure 3)", "run_exposure_analysis.py", ["--chain", "all"]),
        ("4/5: Migration race (Figure 4)", "run_migration_race.py", []),
        ("5/5: Readiness survey (Figure 5)", "run_readiness_survey.py", []),
    ]

    for title, script, extra_args in steps:
        print("=" * 70)
        print(title)
        print("=" * 70)
        result = subprocess.run(
            [sys.executable, str(script_dir / script), "--config", args.config,
             "--output-dir", args.output_dir] + extra_args
        )
        if result.returncode != 0:
            sys.exit(result.returncode)
        print()

    print(f"All results written to {args.output_dir}/")


if __name__ == "__main__":
    main()
