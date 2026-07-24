#!/usr/bin/env python3
"""
Section 8 reproduction: bandwidth-grid widening and the regularized geometric
difference g (Eq. 2) vs. realized out-of-sample gains.

Example:
    python run_bandwidth_geometry_diagnostic.py --config configs/config.yaml --debug
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
from qkernel_finance.features.top8_selector import FactorRotationSelector  # noqa: E402
from qkernel_finance.features.bandwidth import BandwidthScaler  # noqa: E402
from qkernel_finance.quantum.feature_map import QuantumFeatureMap  # noqa: E402
from qkernel_finance.quantum.kernels import FidelityKernel  # noqa: E402
from qkernel_finance.classical.rbf_kernel import ClassicalRBFKernel  # noqa: E402
from qkernel_finance.evaluation.geometry import GeometricDifference  # noqa: E402
from qkernel_finance.evaluation.metrics import PerformanceMetrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bandwidth-grid and geometric-difference diagnostic (Sec 8, Fig. 2).")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="results/", help="Where to write diagnostic outputs")
    parser.add_argument("--debug", action="store_true", help="Run on a small synthetic sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)
    if not args.debug:
        raise SystemExit("Real reproduction needs the paper's actual dataset -- see data/README_data.md. Re-run with --debug for a synthetic smoke test.")

    n = 24
    panel = make_synthetic_panel(num_days=n, num_tickers=n, seed=7)
    characteristics = list(MARKET_CHARACTERISTICS) + list(FUNDAMENTAL_CHARACTERISTICS)
    selector = FactorRotationSelector()
    _active, top8 = selector.select(panel, characteristics, return_col="fwd_return_20d")

    scaler = BandwidthScaler()
    feature_map = QuantumFeatureMap(
        num_qubits=config.get("quantum", "num_qubits", 8),
        repetitions=config.get("quantum", "repetitions_R", 2),
        hq_gate_interpretation=config.get("quantum", "hq_gate_interpretation", "hadamard_conjugated_rz"),
    )
    fidelity_kernel = FidelityKernel()
    rbf_kernel = ClassicalRBFKernel()
    geometry = GeometricDifference()
    metrics = PerformanceMetrics()

    latest_date = sorted(panel["date"].unique())[-1]
    day_df = panel[panel["date"] == latest_date]
    X = day_df[top8].values
    y = day_df["fwd_return_20d"].values

    bandwidth_grid = config.get("quantum", "bandwidth_grid_widened", [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6])
    rows = []
    for lam in bandwidth_grid:
        X_scaled = scaler.scale(X, lam)
        K_q = fidelity_kernel.compute_gram(X_scaled, feature_map)
        K_c = rbf_kernel.compute_gram(X_scaled, gamma=lam)  # reuses lambda as the RBF gamma for this diagnostic

        # Validation IC proxy: correlate the kernel's own row-sums (a crude
        # smoothness/label-alignment proxy) with returns, since this is a
        # single-cross-section smoke test with no separate train/val split.
        ic_q = metrics.rank_ic(K_q.sum(axis=1), y)
        g = geometry.compute(K_c, K_q, lambda_g=config.get("evaluation", "geometry_lambda_g", 1e-6))

        rows.append({"lambda": lam, "validation_ic_proxy": ic_q, "geometric_difference_g": g})
        print(f"lambda={lam:.3f}  IC_proxy={ic_q:.4f}  g={g:.3f}")

    df = pd.DataFrame(rows)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / "bandwidth_geometry_diagnostic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWritten to {out_path}")
    print(
        "\nNOTE: this smoke test uses a single synthetic cross-section and a crude "
        "row-sum IC proxy, not the paper's real validation-IC bandwidth selection "
        "over 60 real windows (Sec 8). It exists to confirm GeometricDifference and "
        "the bandwidth-scaling path run correctly end-to-end, not to reproduce Fig. 2's "
        "actual numbers."
    )


if __name__ == "__main__":
    main()
