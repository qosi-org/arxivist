#!/usr/bin/env python
"""Evaluate a trained checkpoint: Pass@1, Pass@K, Solution Multiplicity on held-out Countdown puzzles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diffsmooth.data.countdown_dataset import CountdownDataset
from diffsmooth.evaluation.metrics import ReasoningMetrics
from diffsmooth.models.policy import PolicyModel
from diffsmooth.rewards.countdown_reward import CountdownVerifier
from diffsmooth.utils.config import Config, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)
    seed_everything(cfg.data.get("seed", 42))

    policy = PolicyModel(
        checkpoint=args.checkpoint,
        use_lora=False,  # checkpoint already has adapters merged/saved
        load_in_4bit=cfg.model["load_in_4bit"],
        device=cfg.hardware["device"],
    )
    verifier = CountdownVerifier()
    metrics_fn = ReasoningMetrics(verifier)

    eval_set = CountdownDataset(
        size=cfg.data["eval_size"],
        num_range=cfg.data["num_range"],
        num_operands=cfg.data["num_operands"],
        seed=cfg.data.get("seed", 42) + 1,  # different seed than training set
    )

    max_k = max(cfg.evaluation["k_values"])
    all_completions = [
        policy.generate([ex["prompt"]], max_k, cfg.training["temperature"], cfg.training["max_new_tokens"])[0]
        for ex in eval_set
    ]

    results = metrics_fn.evaluate_dataset(list(eval_set), all_completions, cfg.evaluation["k_values"])
    print(json.dumps(results, indent=2))

    out_path = Path("results") / f"eval_{Path(args.checkpoint).name}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
