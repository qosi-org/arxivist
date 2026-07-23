"""
2-stage learning rate grid search for the GSL experiment.

Implements the exact 2-stage grid search described in Appendix E.2:
  Stage 1 (coarse): {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
  Stage 2 (refined): centered around best coarse LR

Runs n_seeds=10 models at the best LR and computes aggregate statistics.

Usage:
    python scripts/run_gsl_grid.py --parameterization natural --theta_star 0.5
    python scripts/run_gsl_grid.py --parameterization softmax  --theta_star 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="GSL 2-stage LR grid search (Appendix E.2)")
    p.add_argument("--parameterization",
                   choices=["natural", "sigmoid", "softmax", "sparsemax"],
                   default="natural")
    p.add_argument("--theta_star", type=float, default=0.5)
    p.add_argument("--config",     type=str,   default="configs/gsl.yaml")
    p.add_argument("--output_dir", type=str,   default="results/gsl/")
    p.add_argument("--n_seeds",    type=int,   default=10)
    p.add_argument("--device",     type=str,   default="cuda")
    return p.parse_args()


def run_single(config, device, seed):
    """Train one model and return best val ES."""
    from src.catnat.models.gsl import GSLModel
    from src.catnat.training.trainers import GSLTrainer
    from src.catnat.data.gsl_dataset import get_gsl_dataloaders
    from src.catnat.utils.reproducibility import set_seed

    set_seed(seed)
    model = GSLModel(config)
    trainer = GSLTrainer(model, config, device)
    train_l, val_l, _ = get_gsl_dataloaders(config)
    history = trainer.train(train_l, val_l)
    return history["best_val_es"]


def main():
    args = parse_args()

    from src.catnat.utils.config import load_config
    from src.catnat.utils.reproducibility import get_device

    config = load_config(args.config)
    config.parameterization = args.parameterization
    config.data.theta_star  = args.theta_star
    device = get_device(args.device)

    tc = config.training
    is_natural = args.parameterization == "natural"

    print(f"\n{'='*60}")
    print(f"GSL LR Grid Search")
    print(f"  parameterization: {args.parameterization}")
    print(f"  theta*:           {args.theta_star}")
    print(f"{'='*60}\n")

    # --- Stage 1: Coarse grid ---
    print("Stage 1: Coarse grid")
    coarse_results = {}
    for lr in tc.lr_coarse_grid:
        config.training.lr = lr
        config.training.epochs = 5   # Quick eval
        val_es = run_single(config, device, seed=0)
        coarse_results[lr] = val_es
        print(f"  lr={lr:.4f} → val_ES={val_es:.4f}")

    best_coarse_lr = min(coarse_results, key=coarse_results.get)
    print(f"\nBest coarse lr: {best_coarse_lr} (val_ES={coarse_results[best_coarse_lr]:.4f})")

    # --- Stage 2: Refined grid ---
    refined_grid = tc.lr_refined_natural if is_natural else tc.lr_refined_sigmoid
    print(f"\nStage 2: Refined grid {refined_grid}")
    refined_results = {}
    for lr in refined_grid:
        config.training.lr = lr
        config.training.epochs = 15
        val_es = run_single(config, device, seed=0)
        refined_results[lr] = val_es
        print(f"  lr={lr:.5f} → val_ES={val_es:.4f}")

    best_lr = min(refined_results, key=refined_results.get)
    print(f"\nBest refined lr: {best_lr} (val_ES={refined_results[best_lr]:.4f})")

    # --- Final run: n_seeds models at best LR ---
    config.training.lr = best_lr
    config.training.epochs = tc.epochs   # Full training
    print(f"\nFinal run: {args.n_seeds} seeds at lr={best_lr}")

    from src.catnat.models.gsl import GSLModel
    from src.catnat.training.trainers import GSLTrainer
    from src.catnat.data.gsl_dataset import get_gsl_dataloaders, build_theta_star
    from src.catnat.evaluation.metrics import GSLMetrics
    from src.catnat.utils.reproducibility import set_seed
    import torch

    all_metrics = []
    for seed in range(args.n_seeds):
        set_seed(seed)
        model = GSLModel(config)
        trainer = GSLTrainer(model, config, device)
        train_l, val_l, test_l = get_gsl_dataloaders(config)
        trainer.train(train_l, val_l)

        theta_star_mat = build_theta_star(
            config.model.n_nodes, config.model.n_communities, args.theta_star
        )
        metrics = GSLMetrics.evaluate_all(model, test_l, theta_star_mat, device)
        all_metrics.append(metrics)
        print(f"  Seed {seed}: ES={metrics['es']:.4f}, "
              f"PP-MAE={metrics['pp_mae']:.4f}, theta_MAE={metrics['theta_mae']:.4f}")

    # Aggregate
    import statistics
    summary = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        summary[k] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals)}

    print(f"\n{'='*60}")
    print("Final Results (mean ± std over 10 seeds):")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"{'='*60}")

    # Save results
    out_dir = Path(args.output_dir) / args.parameterization / f"theta{args.theta_star}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"best_lr": best_lr, "metrics": summary}, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
