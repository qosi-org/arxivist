#!/usr/bin/env python
"""
scripts/plot_figures.py
==========================
Reproduce Figures 2-6 from simulated/trained output:
    Fig. 2 — drift/diffusion magnitude temporal profiles
    Fig. 3 — PCA-projected drift/diffusion vector fields
    Fig. 4 — DSLOB regime-degradation curves (needs per-regime metrics)
    Fig. 5 — predicted vs actual scatter plots
    Fig. 6 — ablation training/validation loss curves

Usage:
    python scripts/plot_figures.py --checkpoint checkpoints/artemis_dslob_seed0.pt \
        --dataset dslob --out-dir results/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxivist_artemis.models.artemis import ARTEMIS
from arxivist_artemis.utils.config import Config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce ARTEMIS paper figures.")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="dslob")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--ablation-dir", type=str, default="results/ablation")
    p.add_argument("--out-dir", type=str, default="results/figures")
    return p.parse_args()


def load_artemis(cfg: Config, checkpoint: str, d_x: int, device: str) -> ARTEMIS:
    a = cfg.section("model", "artemis")
    model = ARTEMIS(
        d_x=d_x, d_z=a["d_z"], d_w=a["d_w"], n_lno_poles=a["n_lno_poles"],
        n_fourier=a["n_fourier_frequencies"], drift_hidden_dim=a["drift_hidden_dim"],
        diffusion_hidden_dim=a["diffusion_hidden_dim"],
        aux_pricing_hidden_dim=a["auxiliary_pricing_hidden_dim"],
        n_sde_steps=a["n_sde_steps"], use_symbolic=a["use_symbolic_bottleneck"],
        use_conformal=a["use_conformal_allocation"],
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def plot_drift_diffusion_profiles(model: ARTEMIS, X: torch.Tensor, out_dir: str, device: str) -> None:
    """Reproduce Figure 2: drift/diffusion magnitude across the sequence window."""
    with torch.no_grad():
        eval_times = torch.linspace(0, 1, X.shape[1], device=device)
        t_obs = eval_times.unsqueeze(0).expand(X.shape[0], -1)
        z_enc = model.encoder(X, t_obs, eval_times)
        z0 = z_enc[:, 0, :]
        z_traj = model.sde.simulate(z0, model.n_sde_steps, dt=1.0 / model.n_sde_steps)
        t_grid = torch.linspace(0, 1, model.n_sde_steps + 1, device=device)
        mu_norms, sigma_norms = [], []
        for j in range(model.n_sde_steps + 1):
            zj = z_traj[:, j, :]
            tj = t_grid[j].expand(zj.shape[0])
            mu = model.sde.drift(zj, tj)
            sigma = model.sde.diffusion(zj, tj)
            mu_norms.append(mu.norm(dim=-1).mean().item())
            sigma_norms.append(sigma.norm(dim=(-2, -1)).mean().item())

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    t_axis = np.linspace(0, 1, model.n_sde_steps + 1)
    axes[0].plot(t_axis, mu_norms, label="||mu(Z,t)||")
    axes[0].set_ylabel("Drift magnitude")
    axes[0].legend()
    axes[1].plot(t_axis, sigma_norms, color="tab:red", label="||sigma(Z,t)||")
    axes[1].set_ylabel("Diffusion magnitude")
    axes[1].set_xlabel("Normalised time t in [0,1]")
    axes[1].legend()
    fig.suptitle("ARTEMIS SDE: Drift & Diffusion Temporal Profiles")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure2_drift_diffusion_profiles.png"), dpi=150)
    plt.close(fig)


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str) -> None:
    """Reproduce Figure 5: predicted vs actual scatter."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.4)
    lim = max(np.abs(y_true).max(), np.abs(y_pred).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("ARTEMIS — Prediction Scatter")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure5_prediction_scatter.png"), dpi=150)
    plt.close(fig)


def plot_ablation_losses(ablation_dir: str, out_dir: str) -> None:
    """Reproduce Figure 6: ablation training loss curves, if ablation results exist."""
    variants = ["A0_Full", "A1_NoSDE", "A2_NoPDE", "A3_NoMPR", "A4_NoPhysics", "A5_NoConsistency", "A6_MLP"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()
    any_found = False
    for i, variant in enumerate(variants):
        candidates = [f for f in os.listdir(ablation_dir) if f.startswith(variant) and f.endswith("_losses.npz")] \
            if os.path.isdir(ablation_dir) else []
        if not candidates:
            axes[i].set_visible(False)
            continue
        any_found = True
        losses = np.load(os.path.join(ablation_dir, candidates[0]))["losses"]
        axes[i].plot(losses)
        axes[i].set_title(variant)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
    for j in range(len(variants), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("ARTEMIS Ablation Study — Training Loss")
    fig.tight_layout()
    if any_found:
        fig.savefig(os.path.join(out_dir, "figure6_ablation_losses.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = cfg.device()
    os.makedirs(args.out_dir, exist_ok=True)

    path = os.path.join(args.data_dir, f"{args.dataset}.npz")
    if os.path.exists(path):
        blob = np.load(path)
        X = torch.tensor(blob["X"], dtype=torch.float32, device=device)
        y_true = blob["y"]
        model = load_artemis(cfg, args.checkpoint, X.shape[-1], device)

        plot_drift_diffusion_profiles(model, X[:32], args.out_dir, device)

        # BUGFIX (same root cause as scripts/evaluate.py): batch the forward pass to
        # avoid materializing an OOM-inducing [B,M,N,d_z,d_x] kernel tensor for the
        # full dataset in one call.
        eval_batch_size = 32
        y_pred_chunks = []
        with torch.no_grad():
            for i in range(0, X.shape[0], eval_batch_size):
                xb = X[i : i + eval_batch_size]
                eval_times = torch.linspace(0, 1, xb.shape[1], device=device)
                t_obs = eval_times.unsqueeze(0).expand(xb.shape[0], -1)
                y_pred_chunks.append(model.forward_pretrain(xb, t_obs, eval_times)["y_hat"].cpu().numpy())
        y_pred = np.concatenate(y_pred_chunks)
        plot_prediction_scatter(y_true, y_pred, args.out_dir)
    else:
        print(f"No processed data at {path}; skipping data-dependent figures (2, 5).")

    plot_ablation_losses(args.ablation_dir, args.out_dir)
    print(f"Figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
