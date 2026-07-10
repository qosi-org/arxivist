#!/usr/bin/env python
"""train.py -- Train the LN (or DR) actor-critic policy for a given market
and position-size configuration.

Paper reference: Algorithm 1, Section 6.3.

Usage:
    python train.py --config configs/noise_20lots.yaml
    python train.py --config configs/noise_20lots.yaml --algorithm DR --seed 1
    python train.py --config configs/noise_20lots.yaml --debug --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch

from rlte.env.execution_env import MarketConfig
from rlte.training.trainer import ActorCriticTrainer, TrainConfig
from rlte.utils.config import Config, set_seed


def build_market_config(cfg: Config, lots_override: int | None = None) -> MarketConfig:
    d = cfg["data"]
    return MarketConfig(
        market_type=d["market_type"],
        D=d["D_levels"],
        T=d["T_seconds"],
        dt=d["dt_seconds"],
        start_offset=d["start_offset_seconds"],
        init_bid=d["init_bid_price"],
        init_ask=d["init_ask_price"],
        K=cfg["model"]["K"],
        M=lots_override or d["initial_lots"],
    )


def build_train_config(cfg: Config, debug: bool) -> TrainConfig:
    t = cfg["training"]
    num_envs = 4 if debug else t["num_envs"]
    H = 5 if debug else t["H_iterations"]
    return TrainConfig(
        num_envs=num_envs,
        traj_per_env=t["traj_per_env"],
        N=t["N_steps"],
        H=H,
        learning_rate=t["learning_rate"],
        adam_beta1=t["adam_beta1"],
        adam_beta2=t["adam_beta2"],
        sigma_init=t["sigma_init"],
        sigma_final=t["sigma_final"],
        K=cfg["model"]["K"],
        log_every=t["log_every"],
        checkpoint_every=t["checkpoint_every"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL trade execution policy.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--algorithm", type=str, default=None, help="'LN' or 'DR' (overrides config)")
    parser.add_argument("--market", type=str, default=None, help="Market type override")
    parser.add_argument("--lots", type=int, default=None, help="Initial inventory override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--output-dir", type=str, default="runs/", help="Where to save checkpoints/logs")
    parser.add_argument("--debug", action="store_true", help="Reduced envs/iterations for quick local testing")
    parser.add_argument("--dry-run", action="store_true", help="Build components but do not train")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    seed = args.seed if args.seed is not None else cfg["hardware"].get("seed", 0)
    set_seed(seed, deterministic=cfg["hardware"].get("deterministic", False))

    algorithm = args.algorithm or cfg["training"]["algorithm"]
    if algorithm != "LN":
        raise NotImplementedError(
            "This entrypoint scaffold wires up the LN trainer end-to-end; the "
            "DR (Dirichlet) trainer follows the identical Algorithm 1 loop but "
            "swapping LogisticNormalPolicy for DirichletPolicy -- see "
            "src/rlte/models/policy.py:DirichletPolicy and adapt trainer.py "
            "accordingly (STUB, not wired into ActorCriticTrainer in this scaffold)."
        )

    market_cfg = build_market_config(cfg, lots_override=args.lots)
    if args.market:
        market_cfg.market_type = args.market
    train_cfg = build_train_config(cfg, debug=args.debug)

    trainer = ActorCriticTrainer(market_cfg, train_cfg,
                                  device=cfg["hardware"].get("device", "cpu"))
    print(trainer.summary())
    print(f"Market: {market_cfg.market_type} | Lots: {market_cfg.M} | Seed: {seed}")

    if args.resume:
        state = torch.load(args.resume, map_location=trainer.device)
        trainer.policy.load_state_dict(state["policy"])
        trainer.value.load_state_dict(state["value"])
        print(f"Resumed from {args.resume}")

    if args.dry_run:
        print("Dry run complete: environment and networks constructed successfully.")
        return

    result = trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "final_checkpoint.pt")
    torch.save({"policy": trainer.policy.state_dict(),
                "value": trainer.value.state_dict(),
                "history": result["history"]}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
