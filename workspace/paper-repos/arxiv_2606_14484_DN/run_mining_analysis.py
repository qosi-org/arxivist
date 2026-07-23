#!/usr/bin/env python3
"""
run_mining_analysis.py -- Run the mining-competitiveness model and reproduce
Figure 1 (arXiv:2606.14484, Section 2.2).

Example:
    python run_mining_analysis.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantum_horizon.mining import MiningCompetitivenessModel
from quantum_horizon.utils import load_config, plot_figure1_hashrate, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to write outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.hardware["seed"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mining_cfg = cfg.mining
    model = MiningCompetitivenessModel(
        calibration_hashrate_ghs=mining_cfg["calibration_benchmark_hashrate_ghs"],
        calibration_gate_speed_mhz=mining_cfg["calibration_benchmark_gate_speed_mhz"],
    )

    gate_speeds = np.logspace(
        np.log10(mining_cfg["gate_speed_sweep_ghz_min"]),
        np.log10(mining_cfg["gate_speed_sweep_ghz_max"]),
        mining_cfg["gate_speed_sweep_n_points"],
    )
    hashrates = model.effective_hashrate(gate_speeds)

    hashrate_100ghz = float(model.effective_hashrate(100))
    print(f"Effective hashrate at 100 GHz: {hashrate_100ghz:.2f} TH/s  (paper: ~21 TH/s)")
    print(f"Single ASIC: {mining_cfg['asic_hashrate_ths']} TH/s")
    print(f"2026 network: {mining_cfg['network_hashrate_ehs']} EH/s")

    k_51 = model.machines_for_51_percent(100, mining_cfg["network_hashrate_ehs"])
    print(f"\nMachines needed for 51% of network @ 100GHz: {k_51:.2e}  (paper: ~7e13)")

    summary = {
        "hashrate_at_100ghz_ths": hashrate_100ghz,
        "asic_hashrate_ths": mining_cfg["asic_hashrate_ths"],
        "network_hashrate_ehs": mining_cfg["network_hashrate_ehs"],
        "machines_for_51_percent": k_51,
    }
    with open(output_dir / "mining_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_figure1_hashrate(
        gate_speeds, hashrates, mining_cfg["asic_hashrate_ths"], mining_cfg["network_hashrate_ehs"],
        save_path=str(output_dir / "figure1_hashrate.png"),
    )
    print(f"\nResults written to {output_dir}/")


if __name__ == "__main__":
    main()
