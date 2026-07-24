#!/usr/bin/env python3
"""
Main study entrypoint: runs the walk-forward evaluation (main point-in-time,
170 windows, or diagnostic static-screen, 60 windows) across the kernel-swap
triplet (qkrr-fid, qkrr-pqk, krr-rbf) plus classical benchmarks, and saves
per-window IC series for each model to `--output-dir`.

Real reproduction requires the proprietary China A-share dataset described in
Sec 3.1-3.2 (see data/README_data.md); `--debug` runs against the bundled
synthetic generator instead, for a fast end-to-end pipeline smoke test.

Example:
    python run_study.py --config configs/config.yaml --study main --debug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qkernel_finance.utils.config import Config  # noqa: E402
from qkernel_finance.data.synthetic import make_synthetic_panel  # noqa: E402
from qkernel_finance.data.characteristics import MARKET_CHARACTERISTICS, FUNDAMENTAL_CHARACTERISTICS  # noqa: E402
from qkernel_finance.evaluation.walkforward import WalkForwardEngine  # noqa: E402
from qkernel_finance.features.top8_selector import FactorRotationSelector  # noqa: E402
from qkernel_finance.features.bandwidth import BandwidthScaler  # noqa: E402
from qkernel_finance.quantum.feature_map import QuantumFeatureMap  # noqa: E402
from qkernel_finance.quantum.kernels import FidelityKernel, ProjectedQuantumKernel  # noqa: E402
from qkernel_finance.classical.rbf_kernel import ClassicalRBFKernel  # noqa: E402
from qkernel_finance.classical.baselines import RidgeBaseline  # noqa: E402
from qkernel_finance.models.krr import KernelRidgeRegression  # noqa: E402
from qkernel_finance.evaluation.metrics import PerformanceMetrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the quantum-vs-classical kernel walk-forward study.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--study", type=str, default="main", choices=["main", "diagnostic"], help="Which study to run")
    parser.add_argument("--data-dir", type=str, default="data/", help="Directory with the real characteristic panel (ignored with --debug)")
    parser.add_argument("--output-dir", type=str, default="results/", help="Where to write per-window IC series")
    parser.add_argument("--models", type=str, default="qkrr-fid,krr-rbf,ridge", help="Comma-separated model subset, or 'all'")
    parser.add_argument("--debug", action="store_true", help="Run on a handful of synthetic windows for a quick smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)

    if not args.debug:
        raise SystemExit(
            "Real China A-share data is not bundled with this repo (see data/README_data.md "
            "and architecture_plan.json's risk_assessment: 'High' severity, proprietary data). "
            "Re-run with --debug to smoke-test the full pipeline on synthetic data, or supply "
            "your own data loader conforming to the schema in data/README_data.md."
        )

    num_windows = config.get("debug", "num_windows", 3)
    n_synth = config.get("debug", "subsample_size_N", 32)
    print(f"[--debug] Generating synthetic panel and running {num_windows} windows with subsample N={n_synth}.")

    panel = make_synthetic_panel(num_days=n_synth * 4, num_tickers=max(n_synth, 20), seed=42)
    characteristics = list(MARKET_CHARACTERISTICS) + list(FUNDAMENTAL_CHARACTERISTICS)

    engine = WalkForwardEngine(
        training_window_days=min(60, n_synth * 2),
        rebalance_frequency_days=5,
        cross_section_sample_every_n_days=1,
    )
    selector = FactorRotationSelector(
        active_set_ic_threshold=config.get("feature_selection", "active_set_ic_threshold", 0.015),
        active_set_min_size=config.get("feature_selection", "active_set_min_size", 6),
        top8_size=config.get("feature_selection", "top8_size", 8),
    )
    scaler = BandwidthScaler()
    feature_map = QuantumFeatureMap(
        num_qubits=config.get("quantum", "num_qubits", 8),
        repetitions=config.get("quantum", "repetitions_R", 2),
        hq_gate_interpretation=config.get("quantum", "hq_gate_interpretation", "hadamard_conjugated_rz"),
    )
    fidelity_kernel = FidelityKernel()
    rbf_kernel = ClassicalRBFKernel()
    krr = KernelRidgeRegression()
    metrics = PerformanceMetrics()

    models_requested = args.models.split(",") if args.models != "all" else ["qkrr-fid", "krr-rbf", "ridge"]
    ic_series: dict[str, list[float]] = {m: [] for m in models_requested}

    lam = config.get("quantum", "bandwidth_grid_production", [0.05, 0.1, 0.2, 0.4])[1]  # fixed for the smoke test; real run grid-searches per window
    alpha = config.get("krr", "alpha_grid", [0.001, 0.01, 0.1, 1.0])[1]
    gamma = config.get("quantum", "gamma_projected_kernel_grid", [0.05, 0.1, 0.2, 0.4])[1]

    for window in engine.iter_windows(panel, num_windows=num_windows):
        train_window = window["train_window"]
        rebalance_date = window["rebalance_date"]

        _active_set, top8 = selector.select(train_window, characteristics, return_col="fwd_return_20d")

        train_dates = sorted(train_window["date"].unique())
        fit_df = train_window[train_window["date"] == train_dates[-1]]  # single cross-section as a tiny fit set for the smoke test
        X_train = scaler.scale(fit_df[top8].values, lam)
        y_train = fit_df["fwd_return_20d"].values

        test_df = panel[panel["date"] == rebalance_date]
        if test_df.empty or len(X_train) < 4:
            continue
        X_test = scaler.scale(test_df[top8].values, lam)
        y_test = test_df["fwd_return_20d"].values

        for model_name in models_requested:
            if model_name == "qkrr-fid":
                K_train = fidelity_kernel.compute_gram(X_train, feature_map)
                alpha_hat = krr.fit(K_train, y_train, alpha)
                K_test_train = fidelity_kernel.compute_cross_gram(X_test, X_train, feature_map)
                preds = krr.predict(K_test_train, alpha_hat)
            elif model_name == "qkrr-pqk":
                pqk = ProjectedQuantumKernel()
                phi_train = pqk.bloch_features(X_train, feature_map)
                phi_test = pqk.bloch_features(X_test, feature_map)
                K_train = pqk.compute_gram(phi_train, gamma)
                alpha_hat = krr.fit(K_train, y_train, alpha)
                K_test_train = pqk.compute_cross_gram(phi_test, phi_train, gamma)
                preds = krr.predict(K_test_train, alpha_hat)
            elif model_name == "krr-rbf":
                K_train = rbf_kernel.compute_gram(X_train, gamma)
                alpha_hat = krr.fit(K_train, y_train, alpha)
                K_test_train = rbf_kernel.compute_cross_gram(X_test, X_train, gamma)
                preds = krr.predict(K_test_train, alpha_hat)
            elif model_name == "ridge":
                preds = RidgeBaseline().fit_predict(X_train, y_train, X_test, alpha=config.get("classical_baselines", "ridge_alpha", 10.0))
            else:
                print(f"Skipping unrecognized model '{model_name}' (smoke test supports qkrr-fid, qkrr-pqk, krr-rbf, ridge).")
                continue

            ic = metrics.rank_ic(preds, y_test)
            ic_series[model_name].append(ic)

        print(f"Window {rebalance_date.date()}: " + ", ".join(f"{m}={ic_series[m][-1]:.4f}" for m in models_requested if ic_series[m]))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for model_name, series in ic_series.items():
        if not series:
            continue
        summary = metrics.summarize(np.array(series))
        out_path = Path(args.output_dir) / f"{model_name}_ic_series.csv"
        pd.DataFrame({"ic": series}).to_csv(out_path, index=False)
        print(f"{model_name}: mean_ic={summary['mean_ic']:.4f} icir={summary['icir']:.3f} hit_rate={summary['hit_rate']:.2f} -> {out_path}")

    print(f"Done. Per-window IC series written to {args.output_dir}")


if __name__ == "__main__":
    main()
