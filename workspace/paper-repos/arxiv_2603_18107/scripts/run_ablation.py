#!/usr/bin/env python
"""
scripts/run_ablation.py
==========================
Run one of the 7 DSLOB ablation variants (Table 3: A0_Full .. A6_MLP).

Usage:
    python scripts/run_ablation.py --config configs/config.yaml --variant A2_NoPDE --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxivist_artemis.models.artemis import ARTEMIS
from arxivist_artemis.utils.config import Config

VARIANTS = ["A0_Full", "A1_NoSDE", "A2_NoPDE", "A3_NoMPR", "A4_NoPhysics", "A5_NoConsistency", "A6_MLP"]


class _MLPVariant(nn.Module):
    """A6_MLP: flat MLP on the flattened input window (Section 6.7)."""

    def __init__(self, d_x: int, seq_len: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x * seq_len, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one DSLOB ablation variant.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--variant", type=str, required=True, choices=VARIANTS)
    p.add_argument("--epochs", type=int, default=10, help="Table 3/Fig. 6 ablations use 10 epochs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--out-dir", type=str, default="results/ablation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.set_seed(args.seed)
    device = cfg.device()

    path = os.path.join(args.data_dir, "dslob.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No processed DSLOB data at {path}. Run scripts/prepare_data.py --dataset dslob first.")
    blob = np.load(path)
    X = torch.tensor(blob["X"], dtype=torch.float32, device=device)
    y = torch.tensor(blob["y"], dtype=torch.float32, device=device)
    d_x = X.shape[-1]

    a = cfg.section("model", "artemis")
    lambdas = dict(cfg.section("model", "artemis")["loss_weights"])

    if args.variant == "A6_MLP":
        model = _MLPVariant(d_x, X.shape[1]).to(device)
    else:
        use_symbolic = False  # Table 3 ablations do not include the symbolic layer
        model = ARTEMIS(
            d_x=d_x, d_z=a["d_z"], d_w=a["d_w"], n_lno_poles=a["n_lno_poles"],
            n_fourier=a["n_fourier_frequencies"], drift_hidden_dim=a["drift_hidden_dim"],
            diffusion_hidden_dim=a["diffusion_hidden_dim"],
            aux_pricing_hidden_dim=a["auxiliary_pricing_hidden_dim"],
            n_sde_steps=a["n_sde_steps"], use_symbolic=use_symbolic, use_conformal=False,
        ).to(device)

        if args.variant == "A2_NoPDE":
            lambdas["pde"] = 0.0
        elif args.variant == "A3_NoMPR":
            lambdas["mpr"] = 0.0
        elif args.variant == "A4_NoPhysics":
            lambdas["pde"] = 0.0
            lambdas["mpr"] = 0.0
        elif args.variant == "A5_NoConsistency":
            lambdas["consist"] = 0.0
        # A0_Full: no changes. A1_NoSDE handled specially below (replaces SDE with identity).

    train_cfg = cfg.section("training")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    n = X.shape[0]
    batch_size = train_cfg["batch_size"]

    train_val_losses = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad()

            if args.variant == "A6_MLP":
                y_hat = model(xb)
                loss = torch.nn.functional.mse_loss(y_hat, yb)
            else:
                eval_times = torch.linspace(0, 1, xb.shape[1], device=device)
                t_obs = eval_times.unsqueeze(0).expand(xb.shape[0], -1)
                if args.variant == "A1_NoSDE":
                    # "Simple deterministic transformation" replacing the SDE (Section 6.2):
                    # here implemented as an identity map z_M = z_0 (no stochastic evolution).
                    z_enc = model.encoder(xb, t_obs, eval_times)
                    z0 = z_enc[:, 0, :]
                    y_hat = model.pred_head(z0).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(y_hat, yb)
                else:
                    out = model.forward_pretrain(xb, t_obs, eval_times)
                    losses = model.compute_losses(out["y_hat"], yb, out["z_traj"], out["z_enc"], lambdas)
                    loss = losses["total"]

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
        avg_loss = total_loss / n
        train_val_losses.append(avg_loss)
        print(f"[{args.variant}] epoch {epoch+1}/{args.epochs} loss={avg_loss:.6f}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, f"{args.variant}_seed{args.seed}_losses.npz"),
              losses=np.array(train_val_losses))
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.variant}_seed{args.seed}.pt"))
    print(f"Saved ablation results for {args.variant} to {args.out_dir}/")


if __name__ == "__main__":
    main()
