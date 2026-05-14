#!/usr/bin/env python3
"""
train.py
────────
Main entrypoint for the signature-based implied volatility calibration pipeline.

Runs the full algorithm from Section 4.3:
  1. Generate synthetic market prices (Heston or rough Bergomi MC)
  2. Precompute signatures, Q-matrix, stochastic integrals (offline)
  3. Calibrate ℓ* via L-BFGS-B (online)
  4. Compute IV surfaces and error tables
  5. Save results

Usage:
    python train.py --config configs/config.yaml --experiment heston_uncorr
    python train.py --config configs/config.yaml --experiment rough_bergomi --nMC 100000
    python train.py --config configs/config.yaml --dry-run
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from volsig.utils.config import Config, seed_everything, ensure_dirs
from volsig.utils.plotting import IVSurfacePlotter, print_calibration_summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Signature-based implied volatility calibration (Alòs et al. 2026)"
    )
    p.add_argument("--config", type=str, default="configs/config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--experiment", type=str,
                   choices=["heston_uncorr", "heston_corr", "rough_bergomi"],
                   default="heston_uncorr",
                   help="Which experiment to run (Section 5.1, 5.2, or 6)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory from config")
    p.add_argument("--seed", type=int, default=None,
                   help="Override random seed from config")
    p.add_argument("--nMC", type=int, default=None,
                   help="Override nMC from config")
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                   help="Override device from config")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .npy file with ℓ to resume from")
    p.add_argument("--debug", action="store_true",
                   help="Reduce nMC and maturities for quick local testing")
    p.add_argument("--dry-run", action="store_true",
                   help="Build all components but skip optimisation")
    return p.parse_args()


def apply_overrides(cfg: Config, args) -> Config:
    """Apply CLI overrides to config."""
    if args.seed is not None:
        cfg.simulation.seed = args.seed
    if args.nMC is not None:
        cfg.simulation.nMC = args.nMC
    if args.device is not None:
        cfg.hardware.device = args.device
    if args.output_dir is not None:
        cfg.paths.output_dir = args.output_dir
        cfg.paths.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    if args.debug:
        print("[DEBUG MODE] Reducing nMC=2000 and maturities=[0.1, 0.6]")
        cfg.simulation.nMC = 2_000
        cfg.calibration.maturities = [0.1, 0.6]
        cfg.calibration.max_iter = 20
    return cfg


def patch_experiment(cfg: Config, experiment: str) -> Config:
    """
    Patch config for the selected experiment to match paper parameters.
    Sections 5.1, 5.2, 6 of the paper.
    """
    if experiment == "heston_uncorr":
        # Section 5.1 — uncorrelated Heston (Table 2.1 market params)
        cfg.model.primary_process = "heston_variance"
        cfg.heston_primary.X0 = 0.1
        cfg.heston_primary.nu = 0.2
        cfg.heston_primary.kappa = 2.0
        cfg.heston_primary.theta = 0.15
        cfg.heston_primary.rho_asset_vol = 0.0
        cfg.heston_market.rho = 0.0
        cfg.calibration.maturities = [0.1, 0.6, 1.1, 1.6]

    elif experiment == "heston_corr":
        # Section 5.2 — correlated Heston (Table 2.2 market params)
        cfg.model.primary_process = "heston_variance"
        cfg.heston_primary.X0 = 0.25
        cfg.heston_primary.nu = 0.35
        cfg.heston_primary.kappa = 3.3
        cfg.heston_primary.theta = 0.15
        cfg.heston_primary.rho_asset_vol = -0.5
        cfg.heston_market.rho = -0.5
        cfg.calibration.maturities = [0.1, 0.6, 1.1, 1.6]

    elif experiment == "rough_bergomi":
        # Section 6 — rough Bergomi with shifted-exp fBM primary
        cfg.model.primary_process = "fbm_shifted_exp"
        cfg.fbm_primary.H = 0.2
        cfg.fbm_primary.X0 = 0.1
        cfg.fbm_primary.rho_asset_vol = -0.6
        cfg.calibration.maturities = [0.1, 0.2, 0.4, 0.6]

    return cfg


def generate_market_prices(cfg: Config, experiment: str, output_dir: Path) -> tuple:
    """
    Generate synthetic market option prices and IV surface.
    Returns (prices [nT,nK], iv_surface [nT,nK], iv_analytical [nT,nK]).
    """
    strikes = np.array(cfg.calibration.strikes)
    maturities = np.array(cfg.calibration.maturities)
    nMC_market = min(cfg.simulation.nMC, 200_000)  # market generation uses fewer paths

    if experiment in ("heston_uncorr", "heston_corr"):
        from volsig.models.heston import HestonModel
        m = cfg.heston_market
        market_model = HestonModel(
            sigma0=m.sigma0, nu=m.nu, kappa=m.kappa,
            theta=m.theta, rho=m.rho, S0=cfg.model.S0, r=cfg.model.r,
        )
        print(f"\n[Market] Generating Heston market prices: {market_model}")
        iv_market = market_model.implied_vol_surface_MC(
            strikes, maturities,
            nMC=nMC_market, seed=cfg.simulation.seed + 9999,
        )
        # Analytical benchmark (ASV)
        iv_analytical = market_model.implied_vol_surface_ASV(strikes, maturities)
        analytical_label = "ASV"

    elif experiment == "rough_bergomi":
        from volsig.models.rough_bergomi import RoughBergomiModel
        m = cfg.rbergomi_market
        market_model = RoughBergomiModel(
            sigma0=m.sigma0, H=m.H, eta=m.eta,
            rho=m.rho, S0=cfg.model.S0, r=cfg.model.r,
        )
        print(f"\n[Market] Generating rough Bergomi market prices: {market_model}")
        iv_market = market_model.implied_vol_surface_MC(
            strikes, maturities,
            nMC=nMC_market, seed=cfg.simulation.seed + 9999,
        )
        # VIX analytical benchmark (Section 2.2 calibration)
        from volsig.models.rough_bergomi import RoughBergomiVIXCalibrator
        calibrator = RoughBergomiVIXCalibrator(
            Delta_trading_days=cfg.vix_calibration.Delta_trading_days,
            S0=cfg.model.S0, r=cfg.model.r,
        )
        # Compute VIX ATMI: simulate VIX option at short maturity
        # STUB: VIX option pricing requires nested simulation — using ATM IV as proxy
        # TODO: implement full VIX option pricer (see Risk R6 in architecture plan)
        vix_atmi_proxy = float(np.mean(iv_market[:2, len(strikes)//2]))
        vix_params = calibrator.calibrate(
            iv_surface=iv_market,
            maturities=maturities,
            strikes=strikes,
            vix_atmi=vix_atmi_proxy,
            T_short=float(maturities[0]),
        )
        print(f"[VIX Calibrator] Estimated params: {vix_params}")
        vix_model = RoughBergomiModel(
            sigma0=vix_params["sigma0"], H=vix_params["H"],
            eta=vix_params["eta"], rho=vix_params["rho"],
            S0=cfg.model.S0, r=cfg.model.r,
        )
        iv_analytical = vix_model.implied_vol_surface_MC(
            strikes, maturities, nMC=nMC_market,
            seed=cfg.simulation.seed + 8888,
        )
        analytical_label = "VIX"
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Save market IV
    np.save(output_dir / "iv_market.npy", iv_market)
    np.save(output_dir / "iv_analytical.npy", iv_analytical)
    print(f"[Market] IV saved to {output_dir}")

    return iv_market, iv_analytical, analytical_label


def main():
    args = parse_args()

    # ── Load and patch config ─────────────────────────────────────────────
    cfg = Config.from_yaml(args.config)
    cfg = apply_overrides(cfg, args)
    cfg = patch_experiment(cfg, args.experiment)
    ensure_dirs(cfg)

    output_dir = Path(cfg.paths.output_dir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.simulation.seed)
    print(f"\n{'='*60}")
    print(f"  volsig — Signature IV Calibration")
    print(f"  Experiment: {args.experiment}")
    print(f"  Config:     {args.config}")
    print(f"  nMC:        {cfg.simulation.nMC:,}")
    print(f"  N:          {cfg.model.signature_truncation_N}")
    print(f"  Seed:       {cfg.simulation.seed}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Generate synthetic market prices ──────────────────────────
    iv_market, iv_analytical, analytical_label = generate_market_prices(
        cfg, args.experiment, output_dir
    )

    # ── Step 2: Precompute signatures (offline) ───────────────────────────
    from volsig.models.signature_vol import SignatureVolModel
    sig_model = SignatureVolModel(cfg)
    print(f"\n[Precompute] Building multi-maturity pricer: {sig_model}")
    t_precompute = time.time()
    multi_pricer = sig_model.build_multi_maturity_pricer(
        maturities=cfg.calibration.maturities,
        seed=cfg.simulation.seed,
    )
    t_precompute = time.time() - t_precompute
    print(f"[Precompute] Done in {t_precompute/60:.1f} min")

    if args.dry_run:
        print("\n[dry-run] Precompute succeeded — skipping optimisation.")
        return

    # ── Step 3: Calibrate ℓ* ─────────────────────────────────────────────
    from volsig.calibration.optimizer import SignatureCalibrator

    # Market prices from IV surface (invert BS)
    strikes = np.array(cfg.calibration.strikes)
    maturities = np.array(cfg.calibration.maturities)
    from volsig.pricing.black_scholes import BlackScholes
    sigma0_approx = float(np.nanmean(iv_market))
    market_prices = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            market_prices[i, j] = BlackScholes.call_price(
                cfg.model.S0, K, T, cfg.model.r, iv_market[i, j]
            )

    # Initial ℓ
    n_coords = sig_model.n_coords
    if args.resume:
        l0 = np.load(args.resume)
        print(f"[Calibrate] Resuming from {args.resume}")
    else:
        l0 = None  # zeros (ASSUMED)

    calibrator = SignatureCalibrator(
        market_prices=market_prices,
        strikes=strikes,
        maturities=maturities,
        pricer=multi_pricer,
        S0=cfg.model.S0,
        r=cfg.model.r,
        sigma0=sigma0_approx,
        weight_scheme=cfg.calibration.weight_scheme,
        box_bounds=tuple(cfg.calibration.box_bounds),
        max_iter=cfg.calibration.max_iter,
        tol=cfg.calibration.tolerance,
    )

    t_calib = time.time()
    result = calibrator.calibrate(l0=l0)
    t_calib = time.time() - t_calib

    l_star = result.x
    loss_final = result.fun

    # ── Step 4: Evaluate and report ───────────────────────────────────────
    print_calibration_summary(l_star, loss_final, t_calib, args.experiment)

    iv_sig = multi_pricer.implied_vol_surface(l_star)

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

    # ── Step 5: Save results ──────────────────────────────────────────────
    np.save(output_dir / "l_star.npy", l_star)
    np.save(output_dir / "iv_sig.npy", iv_sig)

    results_meta = {
        "experiment": args.experiment,
        "loss_final": float(loss_final),
        "elapsed_calibration_seconds": float(t_calib),
        "elapsed_precompute_seconds": float(t_precompute),
        "nMC": cfg.simulation.nMC,
        "N": cfg.model.signature_truncation_N,
        "n_coords": int(n_coords),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "l_star": l_star.tolist(),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_meta, f, indent=2)

    # Plot
    plotter.plot_surface_comparison(
        iv1=iv_sig, iv2=iv_analytical,
        strikes=strikes, maturities=maturities,
        labels=["SIG IV", f"{analytical_label} IV"],
        title=f"Implied Volatility Comparison — {args.experiment}",
        save_path=str(output_dir / "iv_surface_comparison.png"),
        show=False,
    )

    print(f"\n[Done] Results saved to {output_dir}/")
    print(f"  l_star.npy, iv_sig.npy, iv_market.npy, iv_analytical.npy")
    print(f"  error_table.csv, results.json, iv_surface_comparison.png\n")


if __name__ == "__main__":
    main()
