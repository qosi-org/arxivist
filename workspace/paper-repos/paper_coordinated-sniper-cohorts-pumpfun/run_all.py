#!/usr/bin/env python3
"""
run_all.py — Master entrypoint: detect → analyze → causal.

Reproduces all paper artifacts in sequence.

Usage:
    python run_all.py \\
        --buyers data/pumpfun_buyers.jsonl \\
        --launches data/pumpfun_launches.jsonl

Paper: Kamat (2026), full reproduction.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full RED-COHORT-2026 reproduction pipeline.")
    p.add_argument("--buyers", type=str, required=True)
    p.add_argument("--launches", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output-dir", type=str, default="results/")
    p.add_argument("--calibrate", action="store_true",
                   help="Calibrate tau to match paper's cohort count")
    p.add_argument("--skip-placebo", action="store_true")
    return p.parse_args()


def run(cmd: list) -> None:
    print(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"[run_all] Step FAILED: {cmd}")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    cohorts_path = str(Path(args.output_dir) / "sniper_cohorts.jsonl")

    steps = [
        ["python", "detect.py",
         "--buyers", args.buyers,
         "--config", args.config,
         "--output", cohorts_path]
        + (["--calibrate"] if args.calibrate else []),

        ["python", "analyze.py",
         "--cohorts", cohorts_path,
         "--output-dir", args.output_dir],

        ["python", "causal.py",
         "--buyers", args.buyers,
         "--launches", args.launches,
         "--cohorts", cohorts_path,
         "--config", args.config]
        + (["--skip-placebo"] if args.skip_placebo else []),
    ]

    for step in steps:
        run(step)

    print(f"\n[run_all] All steps complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
