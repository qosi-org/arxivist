#!/usr/bin/env python
"""Train vanilla-GRPO or DS-GRPO on the Countdown task.

Run both for a real comparison:
    python train.py --config configs/config.yaml --use_differential_smoothing false  # baseline
    python train.py --config configs/config.yaml --use_differential_smoothing true   # paper's fix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diffsmooth.data.countdown_dataset import CountdownDataset
from diffsmooth.models.policy import PolicyModel
from diffsmooth.rewards.countdown_reward import CountdownVerifier
from diffsmooth.training.trainer import DSGRPOTrainer
from diffsmooth.utils.config import Config, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--use_differential_smoothing", type=str, default=None,
                    help="override config's use_differential_smoothing (true/false)")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="tiny run: 2 steps, batch size 2")
    p.add_argument("--dry-run", action="store_true", help="build components, skip training")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)

    seed = args.seed if args.seed is not None else cfg.data.get("seed", 42)
    seed_everything(seed)

    if args.use_differential_smoothing is not None:
        cfg.training["use_differential_smoothing"] = args.use_differential_smoothing.lower() == "true"

    num_steps = 2 if args.debug else cfg.training["num_train_steps"]
    batch_size = 2 if args.debug else cfg.training["batch_size"]

    print("=" * 60)
    print(f"Mode: {'DS-GRPO' if cfg.training['use_differential_smoothing'] else 'vanilla GRPO (baseline)'}")
    print(f"Model: {cfg.model['base_checkpoint']}")
    print(f"Steps: {num_steps}, batch_size: {batch_size}, seed: {seed}")
    print("=" * 60)

    policy = PolicyModel(
        checkpoint=cfg.model["base_checkpoint"],
        use_lora=cfg.model["use_lora"],
        lora_r=cfg.model["lora_r"],
        lora_alpha=cfg.model["lora_alpha"],
        lora_dropout=cfg.model["lora_dropout"],
        load_in_4bit=cfg.model["load_in_4bit"],
        device=cfg.hardware["device"],
    )
    verifier = CountdownVerifier()
    trainer = DSGRPOTrainer(policy, verifier, cfg.training)

    dataset = CountdownDataset(
        size=cfg.data["train_size"],
        num_range=cfg.data["num_range"],
        num_operands=cfg.data["num_operands"],
        seed=seed,
    )
    print(f"Dataset: {len(dataset)} Countdown puzzles (procedurally generated, no download needed)")

    if args.dry_run:
        print("Dry run complete — all components built successfully, skipping training.")
        return

    for step in range(num_steps):
        batch = [dataset[i % len(dataset)] for i in range(step * batch_size, step * batch_size + batch_size)]
        metrics = trainer.train_step(batch)
        if step % 10 == 0 or step == num_steps - 1:
            print(f"step {step:4d} | loss {metrics['loss']:.4f} | mean_reward {metrics['mean_reward']:.3f} "
                  f"| pass@1 {metrics['pass_at_1']:.3f}")

    ckpt_dir = Path("checkpoints") / ("ds_grpo" if cfg.training["use_differential_smoothing"] else "vanilla_grpo")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.model.save_pretrained(ckpt_dir)
    print(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
