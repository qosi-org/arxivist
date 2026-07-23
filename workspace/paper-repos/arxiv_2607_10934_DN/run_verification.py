#!/usr/bin/env python3
"""
run_verification.py — replaces the conventional ArXivist `train.py`.

This paper (arXiv:2607.10934) has no neural network and no dataset to train on, so
there is nothing to "train". Instead, this script runs the Section-5 benchmark
verification suite: for each closed-form case, it simulates the equilibrium and checks
it against the paper's closed-form formulas (terminal covariance identity, terminal
price revelation, empirical MDC health check).

Usage:
    python run_verification.py --config configs/config.yaml --case all
    python run_verification.py --case kyle1985 --debug
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from kyle_liquidity.utils.config import ExperimentConfig
from kyle_liquidity.verification import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        choices=["all", "kyle1985", "back_pedersen1998", "cdf2016", "common_eigenbasis"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Reduce n_steps for a fast smoke test.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build config only, skip full simulation."
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.debug:
        cfg.training["n_steps"] = min(int(cfg.training.get("n_steps", 1000)), 100)
    if args.seed is not None:
        cfg.training["seed"] = args.seed
    used_seed = cfg.seed_everything()

    print(f"[run_verification] paper_id=arxiv_2607_10934 case={args.case} seed={used_seed}")
    print(f"[run_verification] n_steps={cfg.training.get('n_steps')} T={cfg.model.get('T')}")

    if args.dry_run:
        print("[run_verification] --dry-run set: config loaded successfully, skipping simulation.")
        return

    results = run_all(cfg, case=args.case)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "verification_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Verification results ===")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
