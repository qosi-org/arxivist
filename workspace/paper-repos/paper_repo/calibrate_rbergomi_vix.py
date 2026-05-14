#!/usr/bin/env python3
"""
calibrate_rbergomi_vix.py
─────────────────────────
Standalone script to run the VIX-based rough Bergomi calibration.
Implements the new four-step procedure of Section 2.2.

Generates the benchmark IVVIX surface used in Table 6.1.

Usage:
    python calibrate_rbergomi_vix.py --config configs/config.yaml \\
                                     --T1 0.1 --T2 0.6
"""

import argparse
import json
from pathlib import Path

import numpy as np

from volsig.utils.config import Config, seed_everything, ensure_dirs
from volsig.models.rough_bergomi import RoughBergomiModel, RoughBergomiVIXCalibrator
from volsig.utils.plotting import IVSurfacePlotter


def parse_args():
    p = argparse.ArgumentParser(
        description="VIX-based rough Bergomi calibration (Section 2.2)"
    )
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--T1", type=float, default=0.1, help="Short maturity for H estimation")
    p.add_argument("--T2", type=float, default=0.6, help="Long maturity for H estimation")
    p.add_argument("--nMC", type=int, default=100_000)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.simulation.seed)
    ensure_dirs(cfg)

    output_dir = Path(args.output_dir or (Path(cfg.paths.output_dir) / "rough_bergomi" / "vix"))
    output_dir.mkdir(parents=True, exist_ok=True)

    m = cfg.rbergomi_market
    market_model = RoughBergomiModel(
        sigma0=m.sigma0, H=m.H, eta=m.eta, rho=m.rho,
        S0=cfg.model.S0, r=cfg.model.r,
    )
    print(f"\n[VIX Calibration] Market model: {market_model}")
    print(f"  True params: σ₀={m.sigma0}, H={m.H}, η={m.eta}, ρ={m.rho}")

    strikes = np.array(cfg.calibration.strikes)
    maturities = np.array([0.1, 0.2, 0.4, 0.6])

    # Generate market IV
    print("[VIX Calibration] Generating market IV surface by Monte Carlo...")
    iv_market = market_model.implied_vol_surface_MC(
        strikes, maturities, nMC=args.nMC, seed=cfg.simulation.seed
    )

    # Compute VIX ATMI proxy (short-T ATM IV of VIX option)
    # STUB: Full VIX option pricing requires nested simulation (Risk R6).
    # Using short-maturity ATM IV as a proxy — replace with proper VIX simulation.
    # TODO: implement RoughBergomiModel.vix_option_atmi(T_short) for proper calibration.
    vix_atmi_proxy = float(iv_market[0, len(strikes) // 2])
    print(f"[VIX Calibration] VIX ATMI proxy (short T): {vix_atmi_proxy:.6f}")
    print("  WARNING: Using IV proxy for VIX ATMI — replace with proper VIX simulation for paper-accurate results.")

    # Calibration
    calibrator = RoughBergomiVIXCalibrator(
        Delta_trading_days=cfg.vix_calibration.Delta_trading_days,
        S0=cfg.model.S0,
        r=cfg.model.r,
    )

    T1_idx = np.argmin(np.abs(maturities - args.T1))
    T2_idx = np.argmin(np.abs(maturities - args.T2))
    calibrated = calibrator.calibrate(
        iv_surface=iv_market,
        maturities=maturities,
        strikes=strikes,
        vix_atmi=vix_atmi_proxy,
        T_short=float(maturities[T1_idx]),
        T1_idx=int(T1_idx),
        T2_idx=int(T2_idx),
    )
    print("\n[VIX Calibration] Calibrated parameters:")
    true_vals = {"H": m.H, "eta": m.eta, "rho": m.rho, "sigma0": m.sigma0}
    for k, v in calibrated.items():
        print(f"  {k:8s}  true={true_vals[k]:.6f}  calibrated={v:.6f}")

    # Generate IVVIX surface with calibrated params
    vix_model = RoughBergomiModel(
        sigma0=calibrated["sigma0"], H=calibrated["H"],
        eta=calibrated["eta"], rho=calibrated["rho"],
        S0=cfg.model.S0, r=cfg.model.r,
    )
    iv_vix = vix_model.implied_vol_surface_MC(
        strikes, maturities, nMC=args.nMC, seed=cfg.simulation.seed + 1
    )

    # Error table (Table 6.1)
    plotter = IVSurfacePlotter(S0=cfg.model.S0)
    plotter.error_table(
        iv_model=iv_vix,
        iv_market=iv_market,
        iv_analytical=iv_vix,
        strikes=strikes,
        maturities=maturities,
        model_label="VIX",
        analytical_label="MKT",
        save_path=str(output_dir / "vix_error_table.csv"),
    )

    # Save
    np.save(output_dir / "iv_market.npy", iv_market)
    np.save(output_dir / "iv_vix.npy", iv_vix)
    with open(output_dir / "calibrated_params.json", "w") as f:
        json.dump(calibrated, f, indent=2)

    print(f"\n[VIX Calibration] Done. Results saved to {output_dir}/\n")


if __name__ == "__main__":
    main()
