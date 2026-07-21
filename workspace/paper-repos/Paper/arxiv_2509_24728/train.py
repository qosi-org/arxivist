"""
Main training entrypoint for all three experiments.

Usage:
    python train.py --experiment vae --config configs/vae.yaml
    python train.py --experiment gsl --parameterization natural --theta_star 0.5
    python train.py --experiment rl  --config configs/rl.yaml --env BreakoutNoFrameskip-v4

Flags:
    --experiment     {gsl, vae, rl}                  [REQUIRED]
    --config         Path to YAML config file
    --parameterization {natural, sigmoid, softmax, sparsemax}
    --seed           Random seed (overrides config)
    --device         {cuda, cpu}
    --output_dir     Override results/checkpoint dir prefix
    --theta_star     [GSL only] True Bernoulli edge probability
    --N              [VAE only] Number of categorical latent variables
    --K              [VAE only] Categories per variable
    --env            [RL only]  Gymnasium environment ID
    --lr             Override learning rate
    --run_lr_search  Run 2-stage LR grid search before final training
    --debug          Reduce dataset/steps for quick testing
    --dry_run        Build all components but skip training
    --resume         Path to checkpoint to resume from
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train catnat models (arXiv:2509.24728)")
    p.add_argument("--experiment", choices=["gsl", "vae", "rl"], required=True)
    p.add_argument("--config",     type=str, default=None)
    p.add_argument("--parameterization",
                   choices=["natural", "sigmoid", "softmax", "sparsemax"],
                   default=None)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--output_dir", type=str,   default=None)
    p.add_argument("--theta_star", type=float, default=None, help="[GSL] True edge probability")
    p.add_argument("--N",          type=int,   default=None, help="[VAE] Num latent variables")
    p.add_argument("--K",          type=int,   default=None, help="[VAE/RL] Num categories")
    p.add_argument("--env",        type=str,   default=None, help="[RL] Gym env ID")
    p.add_argument("--lr",         type=float, default=None, help="Override learning rate")
    p.add_argument("--run_lr_search", action="store_true",
                   help="Run 2-stage LR grid search (GSL/VAE)")
    p.add_argument("--debug",   action="store_true",
                   help="Reduce data/steps for quick local testing")
    p.add_argument("--dry_run", action="store_true",
                   help="Build all components without training")
    p.add_argument("--resume",  type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load config ---
    from src.catnat.utils.config import load_config
    from src.catnat.utils.reproducibility import set_seed, get_device

    # Default config paths
    default_configs = {"gsl": "configs/gsl.yaml",
                       "vae": "configs/vae.yaml",
                       "rl":  "configs/rl.yaml"}
    config_path = args.config or default_configs[args.experiment]
    config = load_config(config_path)

    # CLI overrides
    if args.parameterization:
        config.parameterization = args.parameterization
    if args.lr:
        config.training.lr = args.lr
    if args.device:
        config.hardware.device = args.device
    if args.output_dir:
        config.output.results_dir    = str(Path(args.output_dir) / "results/")
        config.output.checkpoint_dir = str(Path(args.output_dir) / "checkpoints/")
    if args.experiment == "gsl" and args.theta_star:
        config.data.theta_star = args.theta_star
    if args.experiment == "vae":
        if args.N: config.model.N = args.N
        if args.K:
            config.model.K = args.K
            config.catnat.K = args.K
    if args.experiment == "rl" and args.env:
        config.env.name = args.env

    # Debug mode: shrink dataset
    if args.debug:
        print("[DEBUG] Debug mode: reducing dataset/timesteps.")
        if args.experiment == "gsl":
            config.data.n_samples    = 200
            config.training.epochs   = 2
            config.training.M_samples = 4
        elif args.experiment == "vae":
            config.data.batch_size   = 32
        elif args.experiment == "rl":
            config.training.total_timesteps = 10_000
            config.training.num_steps = 32

    # Seed
    seed = args.seed if args.seed is not None else 0
    set_seed(seed, deterministic=config.hardware.deterministic)
    device = get_device(config.hardware.device)

    print(f"\n[train.py] Experiment: {args.experiment} | "
          f"Parameterization: {config.parameterization} | "
          f"Seed: {seed} | Device: {device}")

    # TensorBoard writer
    log_dir = str(Path(config.output.results_dir) / f"seed_{seed}")
    writer = SummaryWriter(log_dir=log_dir) if not args.dry_run else None

    # --- Dispatch to experiment ---
    if args.experiment == "gsl":
        _train_gsl(config, device, writer, args, seed)
    elif args.experiment == "vae":
        _train_vae(config, device, writer, args, seed)
    elif args.experiment == "rl":
        _train_rl(config, device, writer, args, seed)

    if writer:
        writer.close()
    print("\n[train.py] Done.")


def _train_gsl(config, device, writer, args, seed):
    from src.catnat.models.gsl import GSLModel
    from src.catnat.training.trainers import GSLTrainer
    from src.catnat.data.gsl_dataset import get_gsl_dataloaders

    model = GSLModel(config)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[GSL] Resumed from {args.resume}")

    if args.dry_run:
        print("[GSL] dry_run: model built successfully. Exiting.")
        print(model)
        return

    train_loader, val_loader, test_loader = get_gsl_dataloaders(config)
    trainer = GSLTrainer(model, config, device)

    if args.run_lr_search:
        best_lr = _gsl_lr_search(config, device, seed)
        config.training.lr = best_lr
        trainer._setup_optimizer()
        print(f"[GSL] LR search complete. Using lr={best_lr}")

    history = trainer.train(train_loader, val_loader, writer=writer)
    print(f"\n[GSL] Best val ES: {history['best_val_es']:.4f}")


def _train_vae(config, device, writer, args, seed):
    from src.catnat.models.vae import CatVAE
    from src.catnat.training.trainers import VAETrainer
    from src.catnat.data.mnist_dataset import get_vae_dataloaders

    model = CatVAE(config)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[VAE] Resumed from {args.resume}")

    if args.dry_run:
        print("[VAE] dry_run: model built successfully. Exiting.")
        print(model)
        return

    train_loader, val_loader, test_loader = get_vae_dataloaders(config)
    trainer = VAETrainer(model, config, device)
    history = trainer.train(train_loader, val_loader, writer=writer, n_epochs=100)
    print(f"\n[VAE] Best val ELBO: {history['best_val_elbo']:.2f}")


def _train_rl(config, device, writer, args, seed):
    from src.catnat.models.ppo_agent import PPOAgent
    from src.catnat.training.trainers import PPOTrainer
    from src.catnat.data.atari_wrappers import make_atari_env

    # Create environments
    envs = [
        make_atari_env(config.env.name, seed=seed + i)
        for i in range(config.training.num_envs)
    ]
    n_actions = envs[0].action_space.n

    # Validate K ≥ n_actions and is power of 2
    import math
    K_required = 2 ** math.ceil(math.log2(n_actions))
    if config.catnat.K < n_actions:
        print(f"[RL] WARNING: K={config.catnat.K} < n_actions={n_actions}. "
              f"Setting K={K_required} (next power of 2). See RISK-01.")
        config.catnat.K = K_required

    agent = PPOAgent(config, n_actions=n_actions)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        print(f"[RL] Resumed from {args.resume}")

    if args.dry_run:
        print("[RL] dry_run: agent built successfully. Exiting.")
        print(agent)
        return

    trainer = PPOTrainer(agent, config, device)
    history = trainer.train(envs, writer=writer)

    for env in envs:
        env.close()


def _gsl_lr_search(config, device, seed):
    """Run 2-stage LR grid search for GSL (Appendix E.2)."""
    from src.catnat.models.gsl import GSLModel
    from src.catnat.training.trainers import GSLTrainer
    from src.catnat.data.gsl_dataset import get_gsl_dataloaders
    from src.catnat.utils.reproducibility import set_seed

    tc = config.training
    is_natural = config.parameterization == "natural"
    coarse_grid = tc.lr_coarse_grid

    print(f"[LR Search] Coarse grid: {coarse_grid}")
    best_val, best_lr = float("inf"), coarse_grid[0]

    for lr in coarse_grid:
        set_seed(seed)
        config.training.lr = lr
        model = GSLModel(config)
        trainer = GSLTrainer(model, config, device)
        train_l, val_l, _ = get_gsl_dataloaders(config)
        # Quick eval: 5 epochs only
        config.training.epochs = 5
        history = trainer.train(train_l, val_l)
        config.training.epochs = tc.epochs  # restore
        if history["best_val_es"] < best_val:
            best_val, best_lr = history["best_val_es"], lr
        print(f"  lr={lr:.4f} → val_ES={history['best_val_es']:.4f}")

    # Refined grid
    refined_grid = (tc.lr_refined_natural if is_natural else tc.lr_refined_sigmoid)
    print(f"[LR Search] Refined grid around lr={best_lr}: {refined_grid}")
    for lr in refined_grid:
        set_seed(seed)
        config.training.lr = lr
        model = GSLModel(config)
        trainer = GSLTrainer(model, config, device)
        train_l, val_l, _ = get_gsl_dataloaders(config)
        config.training.epochs = 10
        history = trainer.train(train_l, val_l)
        config.training.epochs = tc.epochs
        if history["best_val_es"] < best_val:
            best_val, best_lr = history["best_val_es"], lr
        print(f"  lr={lr:.5f} → val_ES={history['best_val_es']:.4f}")

    print(f"[LR Search] Best lr={best_lr} (val_ES={best_val:.4f})")
    return best_lr


if __name__ == "__main__":
    main()
