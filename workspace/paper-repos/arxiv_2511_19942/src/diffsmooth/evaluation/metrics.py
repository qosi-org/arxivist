"""Pass@1, Pass@K, and Solution Multiplicity (Eq. 2) — the paper's diversity metric.

Solution Multiplicity SIR ambiguity (confidence 0.55): Eq. 2 references a per-problem quantity
A(x) without an explicit closed-form definition captured from the paper. We implement it as the
count of *distinct* correct completions among K samples for problem x (the most natural reading).
"""
from __future__ import annotations

from diffsmooth.rewards.countdown_reward import CountdownVerifier


class ReasoningMetrics:
    def __init__(self, verifier: CountdownVerifier):
        self.verifier = verifier

    def pass_at_k(self, numbers: list[int], target: int, completions: list[str], k: int) -> float:
        assert k <= len(completions), f"k={k} exceeds number of sampled completions ({len(completions)})"
        subset = completions[:k]
        return 1.0 if any(self.verifier.score(numbers, target, c) > 0 for c in subset) else 0.0

    def solution_multiplicity(self, numbers: list[int], target: int, completions: list[str]) -> float:
        correct = {c.strip() for c in completions if self.verifier.score(numbers, target, c) > 0}
        return float(len(correct))

    def evaluate_dataset(self, examples: list[dict], all_completions: list[list[str]], k_values: list[int]) -> dict:
        results = {f"pass_at_{k}": [] for k in k_values}
        results["solution_multiplicity"] = []

        for ex, completions in zip(examples, all_completions):
            for k in k_values:
                results[f"pass_at_{k}"].append(
                    self.pass_at_k(ex["numbers"], ex["target"], completions, k)
                )
            results["solution_multiplicity"].append(
                self.solution_multiplicity(ex["numbers"], ex["target"], completions)
            )

        return {name: sum(vals) / len(vals) for name, vals in results.items()}
