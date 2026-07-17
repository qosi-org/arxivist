#!/usr/bin/env python
"""
train.py
=========
Standard ArXivist entrypoint name for the "model fitting" stage of this
repo. Unlike a neural-network paper, there is no gradient-based training
loop here: "training" means estimating VAR(1) and Nelson-Siegel parameters
via least squares (Sections 3-5). This is a thin wrapper around
scripts/fit_models.py that adds the standard --resume flag for API
consistency (checkpoints are not applicable to this closed-form estimation
pipeline, so --resume is accepted but has no effect and prints a notice).
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit scenario-generation models (see scripts/fit_models.py).")
    p.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p.add_argument("--resume", type=str, default=None,
                    help="Not applicable: this pipeline uses closed-form OLS/least-squares "
                         "estimation, not iterative training, so there is nothing to resume. "
                         "Accepted for CLI-interface consistency only.")
    p.add_argument("--seed", type=int, default=None, help="Random seed override")
    p.add_argument("--debug", action="store_true", help="Reduced dataset for a quick local test")
    p.add_argument("--dry-run", action="store_true", help="Validate setup without fitting")
    p.add_argument("--freq", type=str, default="daily", choices=["daily", "monthly"])
    p.add_argument("--out-dir", type=str, default="results/fitted_models")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume is not None:
        print(f"[train.py] NOTE: --resume={args.resume} was given but is not applicable "
              "(no iterative training/checkpoints in this closed-form pipeline). Ignoring.")

    cmd = [sys.executable, "scripts/fit_models.py", "--config", args.config,
           "--freq", args.freq, "--out-dir", args.out_dir]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.debug:
        cmd.append("--debug")
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"[train.py] delegating to: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
