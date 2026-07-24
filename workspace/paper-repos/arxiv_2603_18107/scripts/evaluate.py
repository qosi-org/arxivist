#!/usr/bin/env python
"""
scripts/evaluate.py
======================
Compute RMSE/RankIC/DirAcc/Weighted-R2 for a trained checkpoint on a test
split, and (optionally) a Wilcoxon significance test vs. one or more
baseline checkpoints across seeds.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/artemis_dslob_seed0.pt \
        --dataset dslob --out results/metrics_report.md
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxivist_artemis.evaluation.metrics import ForecastMetrics
from arxivist_artemis.evaluation.statistical_tests import SignificanceTester
from arxivist_artemis.models.artemis import ARTEMIS
from arxivist_artemis.utils.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True,
                    choices=["jane_street", "optiver", "time_imm", "dslob"])
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--baseline-scores", type=str, default=None,
                    help="Comma-separated per-seed metric values for a baseline, for Wilcoxon test.")
    p.add_argument("--artemis-scores", type=str, default=None,
                    help="Comma-separated per-seed metric values for ARTEMIS, for Wilcoxon test.")
    p.add_argument("--out", type=str, default="results/metrics_report.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = cfg.device()

    lines = [f"# Evaluation Report — {os.path.basename(args.checkpoint)} on {args.dataset}\n"]

    path = os.path.join(args.data_dir, f"{args.dataset}.npz")
    if os.path.exists(path):
        blob = np.load(path)
        X = torch.tensor(blob["X"], dtype=torch.float32, device=device)
        y_true = blob["y"]
        d_x = X.shape[-1]

        a = cfg.section("model", "artemis")
        model = ARTEMIS(
            d_x=d_x, d_z=a["d_z"], d_w=a["d_w"], n_lno_poles=a["n_lno_poles"],
            n_fourier=a["n_fourier_frequencies"], drift_hidden_dim=a["drift_hidden_dim"],
            diffusion_hidden_dim=a["diffusion_hidden_dim"],
            aux_pricing_hidden_dim=a["auxiliary_pricing_hidden_dim"],
            n_sde_steps=a["n_sde_steps"], use_symbolic=a["use_symbolic_bottleneck"],
            use_conformal=a["use_conformal_allocation"],
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        # BUGFIX (found during Stage 4 validation): running the full dataset through
        # LaplaceNeuralOperator.forward in one pass materializes a [B, M, N, d_z, d_x]
        # kernel tensor that OOMs for realistic dataset sizes, since B is the full
        # dataset size here rather than a mini-batch. Batch the forward pass instead.
        eval_batch_size = 32
        y_pred_chunks = []
        with torch.no_grad():
            for i in range(0, X.shape[0], eval_batch_size):
                xb = X[i : i + eval_batch_size]
                eval_times = torch.linspace(0, 1, xb.shape[1], device=device)
                t_obs = eval_times.unsqueeze(0).expand(xb.shape[0], -1)
                y_pred_chunks.append(model.forward_pretrain(xb, t_obs, eval_times)["y_hat"].cpu().numpy())
        y_pred = np.concatenate(y_pred_chunks)

        lines.append(f"- RMSE: {ForecastMetrics.rmse(y_true, y_pred):.4f}\n")
        lines.append(f"- RankIC: {ForecastMetrics.rank_ic(y_true, y_pred):.4f}\n")
        lines.append(f"- DirAcc: {ForecastMetrics.directional_accuracy(y_true, y_pred):.4f}\n")
        lines.append(f"- Weighted R2: {ForecastMetrics.weighted_r2(y_true, y_pred):.4f}\n")
    else:
        lines.append(f"No processed data found at {path}; run scripts/prepare_data.py first.\n")

    if args.baseline_scores and args.artemis_scores:
        artemis_scores = [float(v) for v in args.artemis_scores.split(",")]
        baseline_scores = [float(v) for v in args.baseline_scores.split(",")]
        result = SignificanceTester.wilcoxon_vs_baseline(artemis_scores, baseline_scores)
        lines.append(f"\n## Wilcoxon signed-rank test vs. baseline\n")
        lines.append(f"- statistic: {result['statistic']:.4f}\n")
        lines.append(f"- p_value: {result['p_value']:.4f} "
                      f"({'significant at alpha=0.01' if result['p_value'] < 0.01 else 'NOT significant at alpha=0.01'})\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.writelines(lines)
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
