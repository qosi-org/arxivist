#!/usr/bin/env python
"""Run the trained model on a single user-provided Countdown puzzle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diffsmooth.data.countdown_dataset import PROMPT_TEMPLATE
from diffsmooth.models.policy import PolicyModel
from diffsmooth.rewards.countdown_reward import CountdownVerifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--numbers", type=str, required=True, help="comma-separated integers, e.g. 4,7,2")
    p.add_argument("--target", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    numbers = [int(n) for n in args.numbers.split(",")]

    policy = PolicyModel(checkpoint=args.checkpoint, use_lora=False, device="cuda")
    verifier = CountdownVerifier()

    prompt = PROMPT_TEMPLATE.format(numbers=numbers, target=args.target)
    completions = policy.generate([prompt], num_samples=1, temperature=0.7, max_new_tokens=256)[0]

    completion = completions[0]
    is_correct = verifier.score(numbers, args.target, completion) > 0

    print(f"Puzzle: numbers={numbers}, target={args.target}")
    print(f"Model output: {completion}")
    print(f"Correct: {is_correct}")


if __name__ == "__main__":
    main()
