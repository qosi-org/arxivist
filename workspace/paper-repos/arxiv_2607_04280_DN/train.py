#!/usr/bin/env python
"""
train.py — run the batch ABM simulation (Sections 2.3-2.4) for one scenario
across N stocks, and save raw trade tapes / price series for later fitting
by evaluate.py.

Despite the name (kept for ArXivist template consistency), this performs no
gradient-based training: it runs a discrete-event agent-based simulation.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --scenario no_splitting
    python train.py --config configs/config.yaml --debug
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqrt_law_abm.data.dataset import StockParameterSampler  # noqa: E402
from sqrt_law_abm.training.trainer import BatchSimulationRunner  # noqa: E402
from sqrt_law_abm.utils.config import Config, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the LOB ABM batch simulation")
    p.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a partially-completed results pickle to resume from (unimplemented "
        "for this discrete-event simulation; kept for CLI-schema consistency)",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        help="Scenario name (baseline, no_splitting, no_hft, price_limits, low_liquidity, "
        "momentum, uniform_split, front_loaded)",
    )
    p.add_argument("--seed", type=int, default=None, help="Global random seed override")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Reduce n_stocks to 4 and n_steps to 5000 for a fast local smoke test",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build all components and validate config without running the simulation",
    )
    p.add_argument(
        "--n-stocks",
        type=int,
        default=None,
        help="Override number of stocks to simulate (defaults to simulation.n_stocks_full)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.debug:
        cfg = cfg.apply_debug_overrides()
        print("[--debug] Using reduced config: n_steps=5000, warmup=500, n_stocks=4")

    seed = args.seed if args.seed is not None else cfg.get("seed", default=42)
    set_global_seed(seed, deterministic=cfg.get("hardware", "deterministic", default=False))

    n_stocks = args.n_stocks or cfg["simulation"]["n_stocks_full"]

    print(f"Config: {args.config}")
    print(f"Scenario: {args.scenario}")
    print(f"Stocks: {n_stocks}")
    print(f"Steps/stock: {cfg['simulation']['n_steps']} (+ {cfg['simulation']['warmup_steps']} warmup)")
    print(f"Seed: {seed}")

    sampler = StockParameterSampler(cfg["model"], cfg["simulation"])
    stock_configs = sampler.sample(n_stocks=n_stocks, seed=seed)

    if args.dry_run:
        print(f"[--dry-run] Sampled {len(stock_configs)} stock configs. Exiting without simulating.")
        return

    runner = BatchSimulationRunner(cfg["model"], cfg["simulation"])
    results = runner.run_all_stocks(
        stock_configs,
        scenario=args.scenario,
        n_jobs=cfg.get("hardware", "num_workers", default=-1),
    )

    out_dir = Path("results") / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "simulation_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    total_trades = sum(len(r.trade_tape) for r in results)
    print(f"Done. {len(results)} stocks simulated, {total_trades} total trades recorded.")
    print(f"Saved to: {out_path}")
    print(f"Next: python evaluate.py --results-dir results/{args.scenario} --scenario {args.scenario}")


if __name__ == "__main__":
    main()
