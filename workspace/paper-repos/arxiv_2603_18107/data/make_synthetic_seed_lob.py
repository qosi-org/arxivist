#!/usr/bin/env python
"""
data/make_synthetic_seed_lob.py
==================================
Generates a placeholder "seed" limit-order-book series so the DSLOB
generation pipeline (Section 3.4: crash-window detection, Vasicek mid-price
fitting, GARCH(1,1) volatility fitting, VAR(1)-correlated noise, time
warping) can be exercised end-to-end for CODE VALIDATION ONLY.

⚠ IMPORTANT: The paper's Section 3.4 states DSLOB is built from "a real
high-frequency limit order book dataset" but never names or provides it.
This script does NOT reproduce that dataset -- it manufactures a plausible
placeholder mid-price series with an embedded synthetic "crash" so that
DSLOBGenerator's detect_crash_window / fit_vasicek / fit_garch /
generate_synthetic_features pipeline has *something* to operate on. Any
results produced downstream from this placeholder are illustrative of code
correctness only and are NOT comparable to the paper's actual DSLOB numbers
(Table 2/3). See data/README_data.md and comparison/hallucination_report.md.

Usage:
    python data/make_synthetic_seed_lob.py --out data/raw/seed_lob.npz --n-obs 5000
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a placeholder seed LOB series for DSLOB pipeline validation.")
    p.add_argument("--out", type=str, default="data/raw/seed_lob.npz")
    p.add_argument("--n-obs", type=int, default=5000)
    p.add_argument("--n-features", type=int, default=85, help="Table 1: DSLOB has 85 features.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = args.n_obs
    # Normal regime: mid price random walk around 100.
    normal_len = int(n * 0.7)
    returns = rng.normal(0.0, 0.001, normal_len)
    normal_prices = 100 * np.cumprod(1 + returns)

    # Embedded "crash": a sharp decline with elevated volatility, mimicking
    # the kind of window the paper's CUSUM/Bayesian change-point detection
    # would identify in a real crash episode.
    crash_len = n - normal_len
    crash_drift = np.linspace(0, -0.15, crash_len)  # cumulative ~15% decline
    crash_returns = rng.normal(0, 0.004, crash_len) + np.diff(np.concatenate([[0], crash_drift]))
    crash_prices = normal_prices[-1] * np.cumprod(1 + crash_returns)

    mid_price = np.concatenate([normal_prices, crash_prices])

    # Remaining 84 features: correlated noise around the mid-price series,
    # loosely mimicking level-specific prices/sizes/spreads (Section 3.4).
    n_extra = args.n_features - 1
    base = np.tile(mid_price[:, None], (1, n_extra))
    scale = rng.uniform(0.01, 1.0, n_extra)
    noise = rng.normal(0, 1, size=(n, n_extra)) * scale
    extra_features = base * 0.01 + noise  # small scale relative to price level, illustrative only

    features = np.column_stack([mid_price, extra_features]).astype(np.float32)

    np.savez(args.out, mid_price=mid_price.astype(np.float32), features=features)
    print(f"Wrote placeholder seed LOB series ({n} obs, {args.n_features} features) to {args.out}")
    print("NOTE: this is a synthetic placeholder, NOT the paper's actual seed dataset. "
          "See data/README_data.md for details.")


if __name__ == "__main__":
    main()
