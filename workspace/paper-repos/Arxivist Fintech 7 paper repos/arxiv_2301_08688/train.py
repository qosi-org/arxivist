"""
train.py — Main Training Entrypoint.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --debug
    python train.py --config configs/config.yaml --resume checkpoints/checkpoint_step50000.pt
    python train.py --config configs/config.yaml --dry-run

For full distributed APEX training (42 workers, 300M steps), RLlib is recommended:
    See configs/config.yaml → use RLlib APEX with framework=torch, num_workers=42

Paper: arXiv:2301.08688
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from apex_lob_trader.utils.config import load_config, set_seeds, get_device
from apex_lob_trader.data.lob_dataset import LOBDataset
from apex_lob_trader.training.environment import LOBTradingEnv
from apex_lob_trader.training.trainer import APEXDQNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train APEX Deep Double Duelling DQN on LOB data (arXiv:2301.08688)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .pt file to resume training from."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run 2000-step debug loop with reduced dataset. Validates setup."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build all components and print summary without training."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Reproducibility
    set_seeds(cfg["seed"], deterministic=cfg.get("hardware", {}).get("deterministic", False))
    device = get_device(cfg)

    # Dataset
    env_cfg = cfg["env"]
    dataset = LOBDataset(
        data_dir=env_cfg["data"]["data_dir"],
        asset=env_cfg["data"]["asset"],
        split="train",
        history_len=env_cfg["history_len"],
    )
    dataset.load()

    # Environment
    env = LOBTradingEnv(dataset=dataset, cfg=cfg)

    # Trainer
    trainer = APEXDQNTrainer(env=env, cfg=cfg, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.dry_run:
        print("\n[Dry Run] All components initialised successfully.")
        print(f"  Environment: {env}")
        print(f"  Main network: {trainer.main_net}")
        print(f"  Replay buffer: {trainer.buffer}")
        print(f"  Device: {device}")
        print("\n[Dry Run] No training performed. Exiting.")
        return

    trainer.train(debug=args.debug)


if __name__ == "__main__":
    main()
