#!/usr/bin/env python3
"""
calibrate_heston_asv.py
───────────────────────
Standalone script to run the Heston analytical ASV expansion calibration.
Implements Section 2.1 (Alòs et al. 2015) — no signature computation required.

Generates the benchmark surfaces used in Tables 5.1, 5.2.

Usage:
    python calibrate_heston_asv.py --config configs/config.yaml \\
                                   --experiment heston_uncorr
"""

import argparse
import json
from pathlib import Path

import numpy as np

from volsig.utils.config import Config, seed_everything, ensure_dirs
from volsig.models.heston import HestonModel
from volsig.utils.plotting import IVSurfacePlotter


def parse_args():
    p = argparse.ArgumentParser(description="Heston ASV analytical calibration (Section 2.1)")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--experiment", type=str,
                   choices=["heston_uncorr", "heston_corr"], default="heston_uncorr")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--nMC", type=int, default=200_000,
                   help="MC paths for market price generation")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.simulation.seed)
    ensure_dirs(cfg)

    output_dir = Path(args.output_dir or (Path(cfg.paths.output_dir) / args.experiment / "asv"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Patch market params
    rho = 0.0 if args.experiment == "heston_uncorr" else -0.5
    m = cfg.heston_market
    market_model = HestonModel(
        sigma0=m.sigma0, nu=m.nu, kappa=m.kappa,
        theta=m.theta, rho=rho,
        S0=cfg.model.S0, r=cfg.model.r,
    )
    print(f"\n[ASV] Market model: {market_model}")

    strikes = np.array(cfg.calibration.strikes)
    maturities = np.array([0.1, 0.6, 1.1, 1.6])

    # Generate market IV by MC
    print("[ASV] Generating market IV surface by Monte Carlo...")
    iv_market = market_model.implied_vol_surface_MC(
        strikes, maturities, nMC=args.nMC, seed=cfg.simulation.seed
    )

    # Analytical ASV surface
    print("[ASV] Computing ASV analytical IV surface...")
    iv_asv = market_model.implied_vol_surface_ASV(strikes, maturities)

    # Calibrate Heston params from ASV system (Section 2.1)
    print("[ASV] Calibrating Heston parameters from IV surface...")
    calibrated = HestonModel.calibrate_from_surface(iv_market, maturities, strikes,
                                                     S0=cfg.model.S0, r=cfg.model.r)
    print("[ASV] Calibrated parameters:")
    for k, v in calibrated.items():
        true_val = getattr(m, k, "n/a")
        print(f"  {k:8s}  true={true_val}  calibrated={v:.6f}")

    # Error table
    plotter = IVSurfacePlotter(S0=cfg.model.S0)
    plotter.error_table(
        iv_model=iv_asv,
        iv_market=iv_market,
        iv_analytical=iv_asv,
        strikes=strikes,
        maturities=maturities,
        model_label="ASV",
        analytical_label="MKT",
        save_path=str(output_dir / "asv_error_table.csv"),
    )

    # Save
    np.save(output_dir / "iv_market.npy", iv_market)
    np.save(output_dir / "iv_asv.npy", iv_asv)
    with open(output_dir / "calibrated_params.json", "w") as f:
        json.dump(calibrated, f, indent=2)

    print(f"\n[ASV] Results saved to {output_dir}/\n")


if __name__ == "__main__":
    main()
