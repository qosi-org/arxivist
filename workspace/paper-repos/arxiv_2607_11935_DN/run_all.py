#!/usr/bin/env python3
"""
run_all.py -- Convenience wrapper running both the observational AIRS
analysis and the six-system simulation validation in sequence, reproducing
Tables 1-3 and Figures 1-3 of arXiv:2607.11935.

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

    print("=" * 70)
    print("STEP 1/2: Observational AIRS regional analysis (Tables 1-2, Figures 1-2)")
    print("=" * 70)
    result1 = subprocess.run(
        [
            sys.executable, str(script_dir / "run_observational_analysis.py"),
            "--config", args.config, "--region", "all", "--output-dir", args.output_dir,
        ]
    )
    if result1.returncode != 0:
        sys.exit(result1.returncode)

    print("\n" + "=" * 70)
    print("STEP 2/2: Simulated tipping-point validation (Table 3, Figure 3)")
    print("=" * 70)
    result2 = subprocess.run(
        [
            sys.executable, str(script_dir / "run_simulation_validation.py"),
            "--config", args.config, "--system", "all", "--output-dir", args.output_dir,
        ]
    )
    if result2.returncode != 0:
        sys.exit(result2.returncode)

    print(f"\nAll results written to {args.output_dir}/")


if __name__ == "__main__":
    main()
