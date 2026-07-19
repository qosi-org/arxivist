#!/usr/bin/env python3
"""
compare_to_paper.py — replaces the conventional ArXivist `evaluate.py`.

Loads outputs/verification_results.json (from run_verification.py) and grades each case
against a simple pass/fail tolerance from configs/config.yaml (evaluation.closed_form_tolerance).
This is a lightweight, repo-local counterpart to Stage 6's fuller comparison report
(see comparison/ for the full ArXivist Stage 6 artifacts).

Usage:
    python compare_to_paper.py --results outputs/verification_results.json
"""
from __future__ import annotations

import argparse
import json
import sys


ERROR_KEYS = [
    "terminal_covariance_identity_residual",
    "terminal_covariance_identity_mc_abs_error",
    "eigen_vs_direct_abs_error_max",
]


def grade(results: dict, tol: float) -> dict:
    graded = {}
    for case, record in results.items():
        errs = {k: v for k, v in record.items() if k in ERROR_KEYS}
        passed = all(v <= tol for v in errs.values()) if errs else None
        graded[case] = {"errors_checked": errs, "passed": passed}
    return graded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=str, default="outputs/verification_results.json")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        with open(args.results) as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"[compare_to_paper] {args.results} not found. Run run_verification.py first.")
        sys.exit(1)

    graded = grade(results, args.tolerance)
    print(json.dumps(graded, indent=2))

    n_pass = sum(1 for v in graded.values() if v["passed"])
    n_checked = sum(1 for v in graded.values() if v["passed"] is not None)
    print(f"\n{n_pass}/{n_checked} cases passed within tolerance={args.tolerance}")


if __name__ == "__main__":
    main()
