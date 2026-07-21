#!/usr/bin/env python3
"""
run_exposure_analysis.py -- Run the Bitcoin and Ethereum exposure-
reconciliation models and the mempool-race model, reproducing Figure 3
(arXiv:2606.14484, Sections 4, 5).

Example:
    python run_exposure_analysis.py --config configs/config.yaml --chain all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_horizon.attacks import MempoolRaceModel
from quantum_horizon.exposure import BitcoinExposureModel, EthereumExposureModel
from quantum_horizon.utils import load_config, plot_figure3_exposure_pie, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--chain", type=str, default="all", choices=["bitcoin", "ethereum", "all"])
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to write outputs")
    return parser.parse_args()


def run_bitcoin(cfg, output_dir: Path) -> dict:
    btc_cfg = cfg.exposure["bitcoin"]
    model = BitcoinExposureModel()

    sources = {
        "glassnode": btc_cfg["glassnode_fraction"],
        "coinbase": btc_cfg["coinbase_btc_millions"] / btc_cfg["total_supply_btc_millions"],
        "coindesk": btc_cfg["coindesk_btc_millions"] / btc_cfg["total_supply_btc_millions"],
        "deloitte_2020": btc_cfg["deloitte_2020_fraction"],
    }
    reconciled_fraction = model.reconcile_sources(sources, method=btc_cfg["reconciliation_method"])
    print("\n=== Bitcoin exposure ===")
    print(f"Reconciled exposed fraction: {reconciled_fraction:.1%}  (paper: ~30%)")

    decomposition = model.decompose(
        btc_cfg["total_supply_btc_millions"], btc_cfg["irreducible_btc_millions"], btc_cfg["migratable_btc_millions"]
    )
    print(f"Irreducible: {decomposition['irreducible_btc']:.1f}M BTC ({decomposition['irreducible_frac']:.1%})")
    print(f"Migratable: {decomposition['migratable_btc']:.1f}M BTC ({decomposition['migratable_frac']:.1%})")
    print(f"Protected: {decomposition['protected_btc']:.1f}M BTC ({decomposition['protected_frac']:.1%})")

    mp = MempoolRaceModel()
    mp_cfg = cfg.mempool_race
    best_case = mp.snipe_success_probability(mp_cfg["fast_clock_derivation_minutes_min"], 0, 1)
    avg_derivation = (mp_cfg["fast_clock_derivation_minutes_min"] + mp_cfg["fast_clock_derivation_minutes_max"]) / 2
    realistic = mp.snipe_success_probability(
        avg_derivation, mp_cfg["realistic_propagation_delay_minutes"], mp_cfg["confirmations_required_realistic"]
    )
    print(f"\nMempool-sniping best-case: {best_case:.1%}  (paper: 41%)")
    print(f"Mempool-sniping realistic: {realistic:.1%}  (paper: ~30%)")

    plot_figure3_exposure_pie(decomposition, save_path=str(output_dir / "figure3_bitcoin_exposure.png"))

    return {
        "reconciled_exposed_fraction": reconciled_fraction,
        "decomposition": decomposition,
        "mempool_best_case": best_case,
        "mempool_realistic": realistic,
    }


def run_ethereum(cfg, output_dir: Path) -> dict:
    eth_cfg = cfg.exposure["ethereum"]
    model = EthereumExposureModel()

    top_down = model.top_down_estimate(eth_cfg["staked_fraction"], eth_cfg["contract_fraction"])
    bottom_up = model.bottom_up_estimate(
        eth_cfg["beacon_overcount_naive_fraction"], eth_cfg["beacon_overcount_correction_factor"]
    )
    low, high = model.reconcile(top_down, bottom_up)

    print("\n=== Ethereum exposure ===")
    print(f"Top-down estimate: {top_down:.1%}  (paper: ~55-63%)")
    print(f"Bottom-up estimate: {bottom_up:.1%}  (paper: ~45-55%)")
    print(f"Reconciled range: {low:.1%} - {high:.1%}  (paper: 50-65%, most defensibly 55-60%)")
    print(f"Dormant-and-exposed: {eth_cfg['dormant_exposed_fraction']:.1%}  (paper: ~0.1%)")

    return {"top_down": top_down, "bottom_up": bottom_up, "range": (low, high)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.hardware["seed"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.chain in ("bitcoin", "all"):
        results["bitcoin"] = run_bitcoin(cfg, output_dir)
    if args.chain in ("ethereum", "all"):
        results["ethereum"] = run_ethereum(cfg, output_dir)

    with open(output_dir / "exposure_analysis_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to {output_dir}/")


if __name__ == "__main__":
    main()
