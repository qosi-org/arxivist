#!/usr/bin/env python
"""
inference.py
=============
Standard ArXivist entrypoint name for single-sample "inference". Here, a
single sample is one simulated scenario trajectory (or yield curve),
generated from previously fitted parameters. This is a thin wrapper around
StationaryBootstrap/VARBootstrap/NelsonSiegelVARBootstrap.simulate() with
n_paths=1, useful for quick sanity checks or interactive exploration.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from arxivist_bootstrap.models.nelson_siegel import NelsonSiegelModel
from arxivist_bootstrap.models.stationary_bootstrap import StationaryBootstrap
from arxivist_bootstrap.models.var_bootstrap import VARBootstrap
from arxivist_bootstrap.models.ns_var_bootstrap import NelsonSiegelVARBootstrap
from arxivist_bootstrap.utils.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a single simulated scenario for quick inspection.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--method", type=str, required=True,
                    choices=["stationary_bootstrap", "var_bootstrap", "ns_var_bootstrap"])
    p.add_argument("--fitted-dir", type=str, default="results/fitted_models")
    p.add_argument("--horizon", type=int, default=60, help="Number of time steps to simulate")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.set_seed(args.seed)

    if args.method == "stationary_bootstrap":
        blob = np.load(os.path.join(args.fitted_dir, "stationary_bootstrap.npz"))
        gen = StationaryBootstrap(n_x=1, n_y=1, seed=args.seed)
        gen.fit(blob["z_hist"])
        out = gen.simulate(1, args.horizon,
                            mean_block_length=cfg.section("model", "stationary_bootstrap")["mean_block_length"])
        print("Simulated z* path (returns, rate diffs), first 5 steps:\n", out[0, :5])

    elif args.method == "var_bootstrap":
        blob = np.load(os.path.join(args.fitted_dir, "var_bootstrap.npz"))
        gen = VARBootstrap(n_x=1, n_y=1, seed=args.seed)
        gen.a0, gen.A1, gen._residuals, gen._fitted = blob["a0"], blob["A1"], blob["residuals"], True
        out = gen.simulate(1, args.horizon, x0=blob["x0"])
        print("Simulated x_hat path (return, rate level), first 5 steps:\n", out[0, :5])

    else:
        blob = np.load(os.path.join(args.fitted_dir, "ns_var_bootstrap.npz"))
        gen = NelsonSiegelVARBootstrap(n_extra=0, seed=args.seed)
        gen.a0, gen.A1, gen._residuals, gen._fitted = blob["a0"], blob["A1"], blob["residuals"], True
        ns_model = NelsonSiegelModel()
        curves = gen.simulate_curves(1, args.horizon, beta0=blob["beta0"],
                                      tau=blob["tau_years"], lam=float(blob["lam"]),
                                      ns_model=ns_model)
        print("Simulated curve at final horizon:\n", curves[0, -1])


if __name__ == "__main__":
    main()
