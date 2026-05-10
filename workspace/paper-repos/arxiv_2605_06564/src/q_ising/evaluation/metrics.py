"""
Policy evaluation utilities for Q-Ising experiments.
Computes mean adoption rate and compares multiple policies.

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from q_ising.data.network import NetworkData
from q_ising.data.sis_simulator import SISSimulator


@dataclass
class EvalResult:
    """Results from evaluating a single policy.

    Attributes:
        policy_name: Identifier string.
        rewards: Per-period adoption rates, shape [n_runs, H].
        mean_reward: Mean adoption rate across periods and runs.
        std_reward: Standard deviation.
    """
    policy_name: str
    rewards: np.ndarray          # [n_runs, H]
    mean_by_period: np.ndarray   # [H] — mean across runs
    std_by_period: np.ndarray    # [H] — std across runs
    mean_reward: float           # scalar — mean across periods and runs
    std_reward: float            # std across period means per run


class PolicyEvaluator:
    """Evaluate and compare policies on a SIS simulator.

    Implements the evaluation protocol from Section 5:
      - H=25 test periods
      - 50 independent test runs
      - Start from zero adoption
      - Metric: mean adoption rate per period

    Args:
        simulator: SISSimulator for the target network.
        network: NetworkData with bin assignments.
    """

    def __init__(self, simulator: SISSimulator, network: NetworkData) -> None:
        self.simulator = simulator
        self.network = network

    def evaluate(
        self,
        policy: Callable,
        policy_name: str,
        H: int = 25,
        n_runs: int = 50,
        seed: int = 0,
    ) -> EvalResult:
        """Evaluate a policy over n_runs independent episodes.

        Args:
            policy: Callable (y [N], t) -> node_action.
            policy_name: Display name.
            H: Test horizon (Section 5: H=25).
            n_runs: Number of independent test episodes (Section 5: 50).
            seed: Base random seed.

        Returns:
            EvalResult with reward statistics.
        """
        rewards = self.simulator.run_test(
            policy=policy,
            H=H,
            n_runs=n_runs,
            seed=seed,
        )  # [n_runs, H]

        mean_by_period = rewards.mean(axis=0)   # [H]
        std_by_period = rewards.std(axis=0)     # [H]

        # Scalar stats: mean and std of per-run total reward
        run_means = rewards.mean(axis=1)        # [n_runs]
        mean_reward = float(run_means.mean())
        std_reward = float(run_means.std())

        return EvalResult(
            policy_name=policy_name,
            rewards=rewards,
            mean_by_period=mean_by_period,
            std_by_period=std_by_period,
            mean_reward=mean_reward,
            std_reward=std_reward,
        )

    def compare_policies(
        self,
        policies: Dict[str, Callable],
        H: int = 25,
        n_runs: int = 50,
        seed: int = 0,
    ) -> Dict[str, EvalResult]:
        """Evaluate multiple policies and return comparison dict.

        Args:
            policies: Dict mapping name -> callable policy.
            H: Test horizon.
            n_runs: Number of test episodes.
            seed: Base seed.

        Returns:
            Dict mapping policy name -> EvalResult.
        """
        results = {}
        for name, policy in policies.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate(policy, name, H=H, n_runs=n_runs, seed=seed)
            print(f"  {name}: {results[name].mean_reward:.4f} ± {results[name].std_reward:.4f}")
        return results

    def results_to_dataframe(self, results: Dict[str, EvalResult]):
        """Convert comparison results to a pandas DataFrame (Table 2 format).

        Args:
            results: Dict from compare_policies().

        Returns:
            pd.DataFrame with columns [policy, mean_reward, std_reward].
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame output")

        rows = []
        for name, r in results.items():
            rows.append({
                "policy": name,
                "mean_reward": r.mean_reward,
                "std_reward": r.std_reward,
                "mean_reward_str": f"{r.mean_reward:.3f} ({r.std_reward:.3f})",
            })
        return pd.DataFrame(rows).set_index("policy")

    def compute_improvement(
        self,
        results: Dict[str, EvalResult],
        baseline: str = "DegreeBin",
    ) -> Dict[str, float]:
        """Compute % improvement of each policy over a baseline (Figure 1 right).

        Args:
            results: Dict from compare_policies().
            baseline: Name of baseline policy.

        Returns:
            Dict mapping policy name -> percentage improvement.
        """
        if baseline not in results:
            raise ValueError(f"Baseline '{baseline}' not in results")
        baseline_mean = results[baseline].mean_reward
        return {
            name: 100.0 * (r.mean_reward - baseline_mean) / max(abs(baseline_mean), 1e-8)
            for name, r in results.items()
        }
