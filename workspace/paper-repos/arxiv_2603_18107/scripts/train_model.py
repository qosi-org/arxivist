#!/usr/bin/env python
"""
scripts/train_model.py
=========================
Train ARTEMIS (Phase 1 pretraining + Phase 2 symbolic distillation,
Algorithm 1) or any one of the 5 baselines (LSTM, Transformer,
NS-Transformer, Informer, Chronos-2) on a chosen dataset.

Usage:
    python scripts/train_model.py --config configs/config.yaml \
        --model artemis --dataset dslob --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxivist_artemis.models.artemis import ARTEMIS
from arxivist_artemis.models.baselines.lstm import LSTMBaseline
from arxivist_artemis.models.baselines.transformer import TransformerBaseline
from arxivist_artemis.models.baselines.ns_transformer import NSTransformerBaseline
from arxivist_artemis.models.baselines.informer import InformerBaseline
from arxivist_artemis.utils.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ARTEMIS or a baseline model.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, required=True,
                    choices=["artemis", "lstm", "transformer", "ns_transformer", "informer", "chronos2"])
    p.add_argument("--dataset", type=str, required=True,
                    choices=["jane_street", "optiver", "time_imm", "dslob"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None,
                    help="Checkpoint path to resume from (loads model + optimizer state).")
    p.add_argument("--debug", action="store_true", help="Tiny-subset quick run for smoke testing.")
    p.add_argument("--dry-run", action="store_true", help="Validate setup without training.")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--epochs", type=int, default=None, help="Override config.training.epochs_pretrain")
    return p.parse_args()


def build_model(model_name: str, d_x: int, cfg: Config) -> torch.nn.Module:
    if model_name == "artemis":
        a = cfg.section("model", "artemis")
        return ARTEMIS(
            d_x=d_x, d_z=a["d_z"], d_w=a["d_w"], n_lno_poles=a["n_lno_poles"],
            n_fourier=a["n_fourier_frequencies"], drift_hidden_dim=a["drift_hidden_dim"],
            diffusion_hidden_dim=a["diffusion_hidden_dim"],
            aux_pricing_hidden_dim=a["auxiliary_pricing_hidden_dim"],
            n_sde_steps=a["n_sde_steps"], use_symbolic=a["use_symbolic_bottleneck"],
            use_conformal=a["use_conformal_allocation"],
        )
    elif model_name == "lstm":
        return LSTMBaseline(d_x=d_x)
    elif model_name == "transformer":
        return TransformerBaseline(d_x=d_x)
    elif model_name == "ns_transformer":
        return NSTransformerBaseline(d_x=d_x)
    elif model_name == "informer":
        return InformerBaseline(d_x=d_x)
    else:
        raise ValueError(
            f"model '{model_name}' requires the Chronos2Baseline wrapper (network access to "
            "Hugging Face Hub); use scripts/train_model.py --model chronos2 only with network available."
        )


def load_dataset(data_dir: str, dataset: str, debug: bool) -> dict:
    """Loads cached windowed tensors produced by scripts/prepare_data.py."""
    path = os.path.join(data_dir, f"{dataset}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No processed data at {path}. Run scripts/prepare_data.py --dataset {dataset} first, "
            "or see data/README_data.md for synthetic-data generation instructions."
        )
    blob = np.load(path)
    X, y, mask = blob["X"], blob["y"], blob["mask"]
    if debug:
        X, y, mask = X[:64], y[:64], mask[:64]
    return {"X": X, "y": y, "mask": mask}


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.set_seed(args.seed)
    device = cfg.device()

    d_x = cfg.section("data", "feature_dims")[args.dataset]

    if args.dry_run:
        model = build_model(args.model, d_x, cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[dry-run] Built model='{args.model}' for dataset='{args.dataset}' "
              f"(d_x={d_x}), {n_params:,} parameters. No training performed.")
        return

    data = load_dataset(args.data_dir, args.dataset, args.debug)
    X = torch.tensor(data["X"], dtype=torch.float32, device=device)
    y = torch.tensor(data["y"], dtype=torch.float32, device=device)
    mask = torch.tensor(data["mask"], dtype=torch.float32, device=device)

    model = build_model(args.model, d_x, cfg).to(device)
    train_cfg = cfg.section("training")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_cfg["lr_reduce_factor"])

    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[train_model.py] Resumed from {args.resume} at epoch {start_epoch}.")

    epochs = args.epochs or train_cfg["epochs_pretrain"]
    batch_size = train_cfg["batch_size"]
    task = "classification" if args.dataset == "dslob" and args.model == "artemis" else "regression"

    n = X.shape[0]
    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb, mb = X[idx], y[idx], mask[idx]
            optimizer.zero_grad()

            if args.model == "artemis":
                eval_times = torch.linspace(0, 1, xb.shape[1], device=device)
                t_obs = torch.linspace(0, 1, xb.shape[1], device=device).unsqueeze(0).expand(xb.shape[0], -1)
                out = model.forward_pretrain(xb, t_obs, eval_times)
                losses = model.compute_losses(
                    out["y_hat"], yb, out["z_traj"], out["z_enc"],
                    cfg.section("model", "artemis")["loss_weights"], task="regression",
                )
                loss = losses["total"]
            else:
                y_hat = model(xb, mb)
                loss = torch.nn.functional.mse_loss(y_hat, yb)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]

        avg_loss = total_loss / n
        scheduler.step(avg_loss)
        print(f"[{args.model}/{args.dataset}] epoch {epoch+1}/{epochs} loss={avg_loss:.6f}")

    if args.model == "artemis" and model.use_symbolic:
        print("[train_model.py] Phase 2: symbolic distillation...")
        for p_ in model.parameters():
            p_.requires_grad_(False)
        for p_ in model.symbolic.parameters():
            p_.requires_grad_(True)
        symb_optimizer = torch.optim.Adam(model.symbolic.parameters(), lr=train_cfg["learning_rate"])
        model.eval()
        with torch.no_grad():
            eval_times = torch.linspace(0, 1, X.shape[1], device=device)
            t_obs = torch.linspace(0, 1, X.shape[1], device=device).unsqueeze(0).expand(X.shape[0], -1)
            teacher_out = model.forward_pretrain(X, t_obs, eval_times)["y_hat"]
        for epoch in range(train_cfg["epochs_symbolic_distill"]):
            symb_optimizer.zero_grad()
            y_symb = model.forward_distill(X)
            lambda_symb = cfg.section("model", "artemis")["loss_weights"]["lambda_symb"]
            loss = torch.nn.functional.mse_loss(y_symb, teacher_out) + model.symbolic.l1_penalty(lambda_symb)
            loss.backward()
            symb_optimizer.step()
            print(f"[symbolic distill] epoch {epoch+1} loss={loss.item():.6f}")
        formula = model.symbolic.export_formula(model.basis_library.feature_names())
        print(f"[symbolic distill] distilled formula: {formula}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_{args.dataset}_seed{args.seed}.pt")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epochs}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
