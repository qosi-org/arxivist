#!/usr/bin/env python
"""
scripts/prepare_data.py
==========================
Run dataset-specific preprocessing (Section 3) and cache windowed tensors
to data/processed/<dataset>.npz for scripts/train_model.py to consume.

Usage:
    python scripts/prepare_data.py --dataset dslob --raw-dir data/raw --out-dir data/processed

NOTE: Jane Street, Optiver, and Time-IMM require the actual Kaggle/Time-IMM
source files (see data/README_data.md). DSLOB's true seed dataset is not
named in the paper; use data/make_synthetic_seed_lob.py to generate a
placeholder seed series for pipeline validation (NOT the paper's actual
DSLOB dataset -- see comparison/hallucination_report.md).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxivist_artemis.data.preprocessing import DSLOBGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess a dataset for ARTEMIS/baseline training.")
    p.add_argument("--dataset", type=str, required=True,
                    choices=["jane_street", "optiver", "time_imm", "dslob"])
    p.add_argument("--raw-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="data/processed")
    p.add_argument("--n-samples", type=int, default=2000, help="For dslob: total synthetic samples to generate")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "dslob":
        seed_path = os.path.join(args.raw_dir, "seed_lob.npz")
        if not os.path.exists(seed_path):
            raise FileNotFoundError(
                f"No seed LOB data at {seed_path}. Run "
                "data/make_synthetic_seed_lob.py first to create a placeholder seed series "
                "(NOTE: this will NOT reproduce the paper's actual DSLOB dataset -- see "
                "data/README_data.md and comparison/hallucination_report.md)."
            )
        seed = np.load(seed_path)
        seed_mid_price, seed_features = seed["mid_price"], seed["features"]

        gen = DSLOBGenerator()
        crash_start, crash_end = gen.detect_crash_window(seed_mid_price)
        crash_mid = seed_mid_price[crash_start:crash_end]
        crash_features = seed_features[crash_start:crash_end]

        vasicek_params = gen.fit_vasicek(crash_mid)
        log_returns = np.diff(np.log(crash_mid + 1e-8))
        garch_params = gen.fit_garch(log_returns)

        synth_mid = gen.simulate_synthetic_mid_price(vasicek_params, args.n_samples, crash_mid[0], args.seed)
        synth_returns = gen.simulate_garch_returns(garch_params, args.n_samples, args.seed)
        synth_features = gen.generate_synthetic_features(crash_features, args.n_samples, args.seed)

        validation = gen.validate_synthetic(crash_features, synth_features)
        print(f"[dslob] validation checks (synthetic vs seed crash window): {validation}")

        # Build windows of length 20 with next-step realized volatility as target
        # (log-transformed square root of sum of squared 1-second log-returns over next 20 steps).
        lookback = 20
        X, y = [], []
        for i in range(lookback, len(synth_features) - lookback):
            X.append(synth_features[i - lookback : i])
            future_returns = synth_returns[i : i + lookback]
            rv = np.sqrt(np.sum(future_returns ** 2))
            y.append(np.log1p(rv))
        X = np.stack(X).astype(np.float32)
        y = np.array(y, dtype=np.float32)
        mask = np.ones_like(X, dtype=np.float32)

        np.savez(os.path.join(args.out_dir, "dslob.npz"), X=X, y=y, mask=mask)
        print(f"[dslob] wrote {X.shape[0]} windows of shape {X.shape[1:]} to "
              f"{args.out_dir}/dslob.npz")
    else:
        raise NotImplementedError(
            f"'{args.dataset}' preprocessing requires the original Kaggle/Time-IMM source files. "
            "See data/README_data.md for obtaining them and the exact loader classes in "
            "src/arxivist_artemis/data/preprocessing.py (JaneStreetPreprocessor, "
            "OptiverPreprocessor, TimeIMMPreprocessor) to build the .npz cache."
        )


if __name__ == "__main__":
    main()
