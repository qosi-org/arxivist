"""Procedurally-generated Countdown puzzles.

STUB NOTE: the paper does not specify its exact Countdown dataset construction (size, generation
algorithm) — see sir.json ambiguities. This generator produces solvable-by-construction puzzles:
pick `num_operands` random integers in `num_range`, apply a random sequence of +-*/ operations to
get a guaranteed-reachable target, matching the general problem format described in the paper.
"""
from __future__ import annotations

import ast
import random

from torch.utils.data import Dataset

from diffsmooth.rewards.countdown_reward import _safe_eval

_OPS = ["+", "-", "*"]

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, each exactly once, write an arithmetic expression "
    "that equals {target}. You may use +, -, *, / and parentheses. "
    "Respond with only the final expression.\n"
)


def _generate_solvable_puzzle(num_operands: int, num_range: tuple[int, int], rng: random.Random):
    numbers = [rng.randint(*num_range) for _ in range(num_operands)]
    expr_numbers = numbers.copy()
    rng.shuffle(expr_numbers)
    expr = str(expr_numbers[0])
    for n in expr_numbers[1:]:
        op = rng.choice(_OPS)
        expr = f"({expr} {op} {n})"
    target = _safe_eval(ast.parse(expr, mode="eval"))
    return numbers, int(target)


class CountdownDataset(Dataset):
    def __init__(self, size: int, num_range: list[int], num_operands: list[int], seed: int = 42):
        rng = random.Random(seed)
        self.examples = []
        for _ in range(size):
            n_ops = rng.randint(num_operands[0], num_operands[1])
            numbers, target = _generate_solvable_puzzle(n_ops, tuple(num_range), rng)
            self.examples.append({"numbers": numbers, "target": target})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = PROMPT_TEMPLATE.format(numbers=ex["numbers"], target=ex["target"])
        return {"numbers": ex["numbers"], "target": ex["target"], "prompt": prompt}
