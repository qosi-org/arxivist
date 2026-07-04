"""
evaluate.py — Evaluation Entrypoint
=====================================
Loads a trained PINN checkpoint and computes the relative L2 error
against a reference solution on a dense spatio-temporal grid.
Optionally generates matplotlib comparison plots.

Usage:
    python evaluate.py --config configs/burgers_continuous.yaml \\
                       --checkpoint results/burgers_continuous/best_model.pt \\
                       --plot

    python evaluate.py --config configs/burgers_discrete.yaml \\
                       --checkpoint results/burgers_discrete/best_model.pt \\
                       --plot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pinns.data.exact_solutions import (
    AllenCahnSpectralSolution,
    BurgersExactSolution,
    SchrodingerSpectralSolution,
)
from pinns.evaluation.metrics import RelativeL2Error
from pinns.models.continuous_pinn import ContinuousPINN
from pinns.models.discrete_pinn import DiscretePINN
from pinns.pde.operators import get_operator
from pinns.utils.config import Config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained PINN")
    p.add_argument("--config",      required=True,  help="YAML config used during training")
    p.add_argument("--checkpoint",  required=True,  help="Path to .pt checkpoint")
    p.add_argument("--reference",   default=None,   help="Path to reference solution .npy (optional)")
    p.add_argument("--output-dir",  default=None,   help="Directory to save plots and metrics")
    p.add_argument("--plot",        action="store_true", help="Generate matplotlib plots")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def evaluate_continuous(
    model: ContinuousPINN,
    cfg: Config,
    device: str,
    reference: str | None,
    output_dir: Path,
    do_plot: bool,
) -> dict:
    pde = cfg.model.pde
    t_range = tuple(cfg.data.t_domain)
    x_range = tuple(cfg.data.x_domain)

    nt = getattr(cfg.evaluation, "grid_t", 100)
    nx = getattr(cfg.evaluation, "grid_x", 256)

    t_grid = np.linspace(*t_range, nt)
    x_grid = np.linspace(*x_range, nx)
    T, X   = np.meshgrid(t_grid, x_grid, indexing="ij")  # [nt, nx]

    t_flat = torch.tensor(T.ravel(), dtype=torch.float32, device=device).unsqueeze(-1)
    x_flat = torch.tensor(X.ravel(), dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        u_pred_flat = model.predict(t_flat, x_flat).cpu().numpy()

    if cfg.model.output_dim == 2:
        # Schrödinger: report |h| = sqrt(u² + v²)
        u_pred = np.sqrt(
            u_pred_flat[:, 0] ** 2 + u_pred_flat[:, 1] ** 2
        ).reshape(nt, nx)
    else:
        u_pred = u_pred_flat.reshape(nt, nx)

    # Reference solution
    if pde == "burgers":
        ref_solver = BurgersExactSolution(nx=nx, nt=nt)
        u_exact = ref_solver.solve()  # [nt, nx]
    elif pde == "schrodinger":
        ref = SchrodingerSpectralSolution(path=reference)
        h_ref = ref.load()  # [nt, nx, 2]
        u_exact = np.sqrt(h_ref[..., 0] ** 2 + h_ref[..., 1] ** 2)
        # Interpolate to evaluation grid if needed
        u_exact = u_exact[:nt, :nx]
    else:
        ref = AllenCahnSpectralSolution(path=reference)
        u_exact = ref.load()[:nt, :nx]

    metric = RelativeL2Error()
    err    = metric.compute(u_pred, u_exact)

    results = {
        "pde": pde,
        "relative_l2_error": err,
        "grid_nt": nt,
        "grid_nx": nx,
    }
    print(f"\n  Relative L2 Error: {err:.4e}")

    target_map = {
        "burgers":     6.7e-4,
        "schrodinger": 1.97e-3,
        "allen_cahn":  6.99e-3,
    }
    target = target_map.get(pde)
    if target:
        factor = err / target
        print(f"  Paper target:      {target:.2e}   (×{factor:.2f})")

    # Save metrics
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if do_plot:
        _plot_continuous(T, X, u_pred, u_exact, pde, output_dir)

    return results


def evaluate_discrete(
    model: DiscretePINN,
    cfg: Config,
    device: str,
    output_dir: Path,
    do_plot: bool,
) -> dict:
    pde     = cfg.model.pde
    t_end   = cfg.model.t_end
    x_range = tuple(cfg.data.x_domain)
    nx      = getattr(cfg.evaluation, "grid_x", 256)
    x_grid  = np.linspace(*x_range, nx)

    x_t = torch.tensor(x_grid, dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    u_pred_np = model.predict_next(x_t).cpu().numpy().ravel()

    # Reference at t_end
    if pde == "burgers":
        ref_solver = BurgersExactSolution(nx=nx, nt=200)
        u_ref_all  = ref_solver.solve()
        t_grid_ref, _ = ref_solver.grid
        t_idx  = int(np.argmin(np.abs(t_grid_ref - t_end)))
        u_exact = u_ref_all[t_idx]
    elif pde == "allen_cahn":
        ref    = AllenCahnSpectralSolution()
        u_all  = ref.load()
        t_grid_ref = np.linspace(0, 1.0, u_all.shape[0])
        t_idx  = int(np.argmin(np.abs(t_grid_ref - t_end)))
        u_exact = u_all[t_idx, :nx]
    else:
        raise ValueError(f"Discrete eval not configured for PDE '{pde}'")

    metric = RelativeL2Error()
    err    = metric.compute(u_pred_np, u_exact)
    results = {"pde": pde, "relative_l2_error": err, "t_end": t_end}

    print(f"\n  Relative L2 Error at t={t_end}: {err:.4e}")
    target_map = {"burgers": 8.2e-4, "allen_cahn": 6.99e-3}
    target = target_map.get(pde)
    if target:
        factor = err / target
        print(f"  Paper target:                   {target:.2e}   (×{factor:.2f})")

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if do_plot:
        _plot_snapshot(x_grid, u_pred_np, u_exact, pde, t_end, output_dir)

    return results


def _plot_continuous(T, X, u_pred, u_exact, pde, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title in zip(axes, [u_pred, u_exact], ["PINN Prediction", "Reference"]):
        im = ax.pcolormesh(T, X, data, shading="auto", cmap="RdBu_r")
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(title)
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"{pde.capitalize()} — Continuous PINN")
    plt.tight_layout()
    path = output_dir / "comparison_heatmap.png"
    plt.savefig(path, dpi=150)
    print(f"  Plot saved to {path}")
    plt.close()


def _plot_snapshot(x_grid, u_pred, u_exact, pde, t_end, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_grid, u_exact, "b-",  lw=2,   label="Exact")
    ax.plot(x_grid, u_pred,  "r--", lw=1.5, label="PINN Prediction")
    ax.set_xlabel("x"); ax.set_ylabel("u(t, x)")
    ax.set_title(f"{pde.capitalize()} — Discrete PINN at t={t_end}")
    ax.legend()
    path = output_dir / f"snapshot_t{t_end}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  Plot saved to {path}")
    plt.close()


def main() -> None:
    args = parse_args()
    cfg  = Config.load(args.config)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pde = cfg.model.pde
    op  = get_operator(pde)
    ckpt = torch.load(args.checkpoint, map_location=device)

    if cfg.model.type == "continuous":
        model = ContinuousPINN(
            hidden_layers=cfg.model.hidden_layers,
            hidden_neurons=cfg.model.hidden_neurons,
            pde_operator=op,
            output_dim=getattr(cfg.model, "output_dim", 1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        evaluate_continuous(model, cfg, device, args.reference, output_dir, args.plot)

    elif cfg.model.type == "discrete":
        q = cfg.model.q
        model = DiscretePINN(
            hidden_layers=cfg.model.hidden_layers,
            hidden_neurons=cfg.model.hidden_neurons,
            q=q,
            pde_operator=op,
            dt=cfg.model.dt,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        evaluate_discrete(model, cfg, device, output_dir, args.plot)


if __name__ == "__main__":
    main()
