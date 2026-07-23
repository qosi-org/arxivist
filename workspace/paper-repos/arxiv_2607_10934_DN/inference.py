#!/usr/bin/env python3
"""
inference.py — single-path demo (analog of ArXivist's conventional inference entrypoint).

Given (C, sigma, T, v_true) from configs/config.yaml, simulates ONE realized equilibrium
path for the scalar Kyle (1985) case and prints price/depth/strategy trajectories at a
handful of checkpoints.

Usage:
    python inference.py --config configs/config.yaml --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from kyle_liquidity.depth import KyleConstantVolDepth
from kyle_liquidity.filtering import EquilibriumSimulator
from kyle_liquidity.strategy import InsiderStrategy
from kyle_liquidity.utils.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.seed_everything(args.seed)

    Sigma_0 = float(cfg.model.get("C", 1.0))
    sigma = float(cfg.model.get("sigma", 1.0))
    T = float(cfg.model.get("T", 1.0))
    n_steps = int(cfg.training.get("n_steps", 1000))

    model = KyleConstantVolDepth(Sigma_0=Sigma_0, sigma=sigma, T=T)
    sim = EquilibriumSimulator(
        n_assets=1,
        M_star_fn=lambda t: np.array([[model.M_star(np.array([t]))[0]]]),
        Sigma_star_fn=lambda t: np.array([[model.Sigma_star(np.array([t]))[0]]]),
        sigma_fn=lambda t: np.array([[sigma]]),
        strategy=InsiderStrategy(),
    )
    v_true = np.array([float(np.sqrt(Sigma_0)) * 1.0])
    path = sim.simulate(v_true=v_true, p0=np.array([0.0]), T=T, n_steps=n_steps, seed=args.seed)

    checkpoints = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    print(f"v_true = {v_true[0]:.4f}, Sigma_0 = {Sigma_0}, sigma = {sigma}, T = {T}")
    print(f"{'t':>8} {'P_t':>10} {'X_t':>10} {'M*_t':>10}")
    for k in checkpoints:
        t = path["t"][k]
        print(f"{t:8.4f} {path['P'][k,0]:10.4f} {path['X'][k,0]:10.4f} {model.M_star(np.array([t]))[0]:10.4f}")


if __name__ == "__main__":
    main()
