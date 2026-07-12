#!/usr/bin/env python
"""
inference.py — run a single-stock simulation end-to-end and print its
fitted delta/c. The fastest way to sanity-check a config or code change
without waiting for a full 2000-stock (or even 20-stock) batch run.

Usage:
    python inference.py --config configs/config.yaml --scenario baseline --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqrt_law_abm.data.dataset import StockParameterSampler, apply_scenario  # noqa: E402
from sqrt_law_abm.data.transforms import MetaorderReconstructor  # noqa: E402
from sqrt_law_abm.evaluation.metrics import ImpactCurveFitter  # noqa: E402
from sqrt_law_abm.models.market import StockMarketSimulation  # noqa: E402
from sqrt_law_abm.utils.config import Config, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one stock and print its fitted delta/c")
    p.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p.add_argument("--scenario", type=str, default="baseline", help="Scenario name")
    p.add_argument("--seed", type=int, default=0, help="Seed for this single stock")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    set_global_seed(args.seed)

    sampler = StockParameterSampler(cfg["model"], cfg["simulation"])
    stock_config = sampler.sample(n_stocks=1, seed=args.seed)[0]
    stock_config = apply_scenario(stock_config, args.scenario)

    print(f"Stock config: {stock_config}")
    sim = StockMarketSimulation(stock_config.to_dict(), cfg["model"], cfg["simulation"])

    n_steps = cfg["simulation"]["n_steps"]
    warmup = cfg["simulation"]["warmup_steps"]
    print(f"Running {n_steps} steps ({warmup} warmup)... this may take a few minutes.")
    result = sim.run(n_steps=n_steps, warmup_steps=warmup)
    print(f"Recorded {len(result.trade_tape)} trades.")

    reconstructor = MetaorderReconstructor()
    daily_volume, daily_range = result.daily_volume_and_range()
    metaorders = reconstructor.reconstruct(
        result.trade_tape,
        delta_t_ticks=cfg["data"]["metaorder_delta_t_ticks"],
        steps_per_day=result.steps_per_day,
    )
    q_norm, i_norm = reconstructor.normalize(
        metaorders, daily_volume, daily_range, cfg["data"]["min_qnorm_threshold_pct"]
    )

    fitter = ImpactCurveFitter()
    try:
        fit_result = fitter.log_bin_and_fit(
            q_norm, i_norm, n_bins=cfg["data"]["log_bins"], min_pts=cfg["data"]["min_points_per_bin"]
        )
        print(f"\nFitted delta = {fit_result['delta']:.3f}, c = {fit_result['c']:.3f}")
        print(f"(from {fit_result['n_meta']} metaorders)")
    except ValueError as e:
        print(f"\nCould not fit impact curve: {e}")
        print("Try a longer simulation (larger simulation.n_steps) to accumulate more metaorders.")


if __name__ == "__main__":
    main()
