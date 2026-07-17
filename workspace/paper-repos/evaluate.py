#!/usr/bin/env python
"""
evaluate.py
============
Standard ArXivist entrypoint name for the evaluation stage. Runs
scripts/simulate.py followed by scripts/evaluate.py to reproduce the
Section 6/8 comparison metrics and Appendix B arbitrage checks for a chosen
methodology.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate + evaluate a chosen methodology end-to-end.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--method", type=str, required=True,
                    choices=["stationary_bootstrap", "var_bootstrap", "ns_var_bootstrap"])
    p.add_argument("--fitted-dir", type=str, default="results/fitted_models")
    p.add_argument("--n-paths", type=int, default=10000)
    p.add_argument("--freq", type=str, default="daily", choices=["daily", "monthly"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sims-out", type=str, default="results/simulations.npz")
    p.add_argument("--report-out", type=str, default="results/metrics_report.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sim_cmd = [
        sys.executable, "scripts/simulate.py",
        "--config", args.config, "--method", args.method,
        "--fitted-dir", args.fitted_dir, "--n-paths", str(args.n_paths),
        "--freq", args.freq, "--seed", str(args.seed), "--out", args.sims_out,
    ]
    print(f"[evaluate.py] delegating to: {' '.join(sim_cmd)}")
    subprocess.run(sim_cmd, check=True)

    eval_cmd = [
        sys.executable, "scripts/evaluate.py",
        "--config", args.config, "--simulations", args.sims_out,
        "--historical", f"{args.fitted_dir}/{args.method}.npz", "--out", args.report_out,
    ]
    print(f"[evaluate.py] delegating to: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
