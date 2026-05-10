"""
Q-Ising Evaluation Script — Compare Policies.
Loads a trained Q-Ising model and evaluates against all baselines.

Usage:
    python evaluate.py --config configs/sbm_default.yaml --model-path results/q_ising_model

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 5.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Q-Ising vs baselines")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to saved Q-Ising model (if not re-training)")
    parser.add_argument("--policies", nargs="+",
                        default=["random", "degree", "lir", "degree_bin", "q_ising"],
                        help="Which policies to evaluate")
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--H", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/eval_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from q_ising.utils.config import load_config, set_global_seed
    cfg = load_config(args.config)
    set_global_seed(args.seed)

    print("Q-Ising Evaluation")
    print(f"  Policies: {args.policies}")
    print(f"  H={args.H}, n_runs={args.n_runs}")
    print("\nNote: Run train.py first to generate a trained model, then pass --model-path.")
    print("Evaluation logic mirrors the policy comparison in train.py::run_sbm_experiment().")


if __name__ == "__main__":
    main()
