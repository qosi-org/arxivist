#!/usr/bin/env python3
"""
evaluate.py
───────────
Evaluate a calibrated ℓ* vector against the market IV surface.
Produces error tables (Tables 5.1, 5.2, 6.1) and IV surface plots.

Usage:
    python evaluate.py --config configs/config.yaml \\
                       --l_star results/heston_uncorr/l_star.npy \\
                       --experiment heston_uncorr \\
                       --plot
"""

import argparse
import json
from pathlib import Path

import numpy as np

from volsig.utils.config import Config, seed_everything
from volsig.utils.plotting import IVSurfacePlotter


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate calibrated ℓ* vector")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--l_star", type=str, required=True,
                   help="Path to .npy file with calibrated ℓ* vector")
    p.add_argument("--experiment", type=str,
                   choices=["heston_uncorr", "heston_corr", "rough_bergomi"],
                   required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--plot", action="store_true", help="Show/save 3D surface plots")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.seed:
        cfg.simulation.seed = args.seed
    seed_everything(cfg.simulation.seed)

    output_dir = Path(args.output_dir or (Path(cfg.paths.output_dir) / args.experiment))
    output_dir.mkdir(parents=True, exist_ok=True)

    l_star = np.load(args.l_star)
    print(f"\n[Evaluate] Loaded ℓ* from {args.l_star}  (n={len(l_star)})")

    # Load pre-saved IV surfaces if available, else recompute
    iv_market_path = output_dir / "iv_market.npy"
    iv_analytical_path = output_dir / "iv_analytical.npy"
    iv_sig_path = output_dir / "iv_sig.npy"

    if iv_market_path.exists():
        iv_market = np.load(iv_market_path)
        iv_analytical = np.load(iv_analytical_path)
        print(f"[Evaluate] Loaded IV surfaces from {output_dir}")
    else:
        raise FileNotFoundError(
            f"IV surfaces not found in {output_dir}. Run train.py first."
        )

    if iv_sig_path.exists():
        iv_sig = np.load(iv_sig_path)
    else:
        # Recompute from ℓ*
        from volsig.models.signature_vol import SignatureVolModel
        from train import patch_experiment
        cfg = patch_experiment(cfg, args.experiment)
        sig_model = SignatureVolModel(cfg)
        multi_pricer = sig_model.build_multi_maturity_pricer()
        iv_sig = multi_pricer.implied_vol_surface(l_star)
        np.save(iv_sig_path, iv_sig)

    strikes = np.array(cfg.calibration.strikes)
    maturities = np.array(cfg.calibration.maturities)
    analytical_label = "ASV" if args.experiment != "rough_bergomi" else "VIX"

    plotter = IVSurfacePlotter(S0=cfg.model.S0)
    plotter.error_table(
        iv_model=iv_sig,
        iv_market=iv_market,
        iv_analytical=iv_analytical,
        strikes=strikes,
        maturities=maturities,
        model_label="SIG",
        analytical_label=analytical_label,
        save_path=str(output_dir / "error_table.csv"),
    )

    if args.plot:
        plotter.plot_surface_comparison(
            iv1=iv_sig, iv2=iv_analytical,
            strikes=strikes, maturities=maturities,
            labels=["SIG IV", f"{analytical_label} IV"],
            title=f"IV Comparison — {args.experiment}",
            save_path=str(output_dir / "iv_comparison.png"),
            show=True,
        )

    print(f"\n[Evaluate] Done. Results saved to {output_dir}/\n")


if __name__ == "__main__":
    main()
