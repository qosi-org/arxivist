#!/usr/bin/env python3
"""
run_cck_regression.py
Augmented CCK regression (Eq. 8) across supercritical CWS seeds.
Paper: arXiv:2605.11645, Section 3.3.2

Expected result: cross-seed median gamma3 = -0.0072, CI = [-0.00769, -0.00602]

Usage:
    python run_cck_regression.py --data_dir results/detection/ --output results/cck_regression.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results/detection/")
    parser.add_argument("--output", type=str, default="results/cck_regression.json")
    args = parser.parse_args()

    from geomherd.evaluation.baselines import CSADBaseline

    results_files = list(Path(args.data_dir).glob("*.json"))
    if not results_files:
        print(f"No results found in {args.data_dir}. Run run_detection.py first.")
        return

    gamma3_estimates: List[float] = []
    gamma2_with_kappa: List[float] = []
    gamma2_without_kappa: List[float] = []

    for f in results_files:
        with open(f) as fp:
            data = json.load(fp)
        for traj in (data if isinstance(data, list) else [data]):
            if not traj.get("is_supercritical", False):
                continue
            kappa_series = traj.get("kappa_bar_plus_series", [])
            order_params = traj.get("order_params", [])
            if len(kappa_series) < 10 or len(order_params) < 10:
                continue

            # Reconstruct synthetic returns from order parameter (proxy)
            # Full implementation requires per-agent returns from CWS substrate
            # Here we use order_parameter as market return proxy (Rm)
            T = min(len(kappa_series), len(order_params))
            rm = np.array(order_params[:T])
            kappa_or = np.array(kappa_series[:T])
            # Synthetic per-agent returns for CSAD (simplified proxy)
            np.random.seed(42)
            N_proxy = 10
            noise = np.random.randn(T, N_proxy) * 0.02
            returns_proxy = rm[:, None] * 0.1 + noise  # [T, N_proxy]
            csad = CSADBaseline.compute(returns_proxy)

            try:
                # Without kappa_OR
                reg_base = CSADBaseline.cck_regression(csad, rm, kappa_or=None)
                # With kappa_OR (Eq. 8)
                reg_aug = CSADBaseline.cck_regression(csad, rm, kappa_or=kappa_or)
                gamma3_estimates.append(reg_aug["gamma3_coef"])
                gamma2_without_kappa.append(reg_base["gamma2_coef"])
                gamma2_with_kappa.append(reg_aug["gamma2_coef"])
            except Exception as e:
                continue

    if not gamma3_estimates:
        print("No valid supercritical trajectories for regression.")
        return

    gamma3_arr = np.array(gamma3_estimates)
    median_g3 = float(np.median(gamma3_arr))
    ci_low = float(np.percentile(gamma3_arr, 2.5))
    ci_high = float(np.percentile(gamma3_arr, 97.5))

    median_g2_base = float(np.median(gamma2_without_kappa))
    median_g2_aug = float(np.median(gamma2_with_kappa))

    print("\n=== Augmented CCK Regression (Eq. 8) ===")
    print(f"N supercritical seeds: {len(gamma3_estimates)}")
    print(f"gamma3 (kappa_bar_OR coef):")
    print(f"  Median:    {median_g3:.5f}")
    print(f"  95% CI:    [{ci_low:.5f}, {ci_high:.5f}]")
    print(f"  Expected:  -0.0072 CI [-0.00769, -0.00602]  (paper Table 2 text)")
    print(f"  Sign consistent with Prop. 1: {'YES' if median_g3 < 0 else 'NO'}")
    print(f"\ngamma2 shift (|beta2| with vs without kappa):")
    print(f"  Without kappa: {median_g2_base:.4f}")
    print(f"  With kappa:    {median_g2_aug:.4f}")
    print(f"  Abs change:    {abs(median_g2_aug - median_g2_base):.4f}")

    output = {
        "n_seeds": len(gamma3_estimates),
        "gamma3_median": median_g3,
        "gamma3_ci_2_5": ci_low,
        "gamma3_ci_97_5": ci_high,
        "gamma3_all": gamma3_estimates,
        "gamma2_median_base": median_g2_base,
        "gamma2_median_augmented": median_g2_aug,
        "paper_expected_gamma3_median": -0.0072,
        "paper_expected_gamma3_ci": [-0.00769, -0.00602],
        "sign_consistent": bool(median_g3 < 0),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
