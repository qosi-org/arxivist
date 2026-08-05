#!/usr/bin/env python
"""
inference.py
=============
Single-sample inference: run a trained ARTEMIS checkpoint on one input
window and print the point prediction, conformal interval, and (if enabled)
the distilled symbolic prediction.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from arxivist_artemis.models.artemis import ARTEMIS
from arxivist_artemis.utils.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ARTEMIS inference on a single input window.")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input-npz", type=str, required=True, help="npz with a single 'x' array [L, d_x]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = cfg.device()

    data = np.load(args.input_npz)
    x = torch.tensor(data["x"], dtype=torch.float32, device=device).unsqueeze(0)  # [1, L, d_x]
    d_x = x.shape[-1]

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

    with torch.no_grad():
        eval_times = torch.linspace(0, 1, x.shape[1], device=device)
        t_obs = eval_times.unsqueeze(0)
        y_hat = model.forward_pretrain(x, t_obs, eval_times)["y_hat"].item()

    print(f"Point prediction: {y_hat:.6f}")
    if model.use_conformal:
        lo, hi = model.conformal.predict_interval(y_hat)
        print(f"Conformal interval (alpha={model.conformal.alpha}): [{lo:.6f}, {hi:.6f}]")
    if model.use_symbolic:
        with torch.no_grad():
            y_symb = model.forward_distill(x).item()
        print(f"Symbolic-distilled prediction: {y_symb:.6f}")


if __name__ == "__main__":
    main()
