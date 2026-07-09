#!/usr/bin/env python
"""evaluate.py -- Evaluate a trained (or heuristic) policy over Monte Carlo
market simulations and reproduce Table 1 / Figure 4 style outputs.

Paper reference: Section 6.3, Table 1.

Usage:
    python evaluate.py --policy SL --market noise --lots 20
    python evaluate.py --policy LN --checkpoint runs/final_checkpoint.pt --market noise --lots 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch

from rlte.env.execution_env import MarketConfig
from rlte.evaluation.benchmarks import SubmitAndLeave, TWAP
from rlte.evaluation.metrics import Evaluator
from rlte.models.policy import LogisticNormalPolicy, DirichletPolicy
from rlte.utils.config import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trade execution policy.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained policy checkpoint")
    parser.add_argument("--policy", type=str, default="LN", help="'LN' | 'DR' | 'SL' | 'TWAP'")
    parser.add_argument("--market", type=str, default="noise",
                         help="noise | noise_tactical | noise_tactical_strategic")
    parser.add_argument("--lots", type=int, default=20)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--num-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1, help="Eval seed, distinct from training")
    parser.add_argument("--output", type=str, default="results/eval_result.json")
    args = parser.parse_args()

    set_seed(args.seed)
    market_cfg = MarketConfig(market_type=args.market, K=args.K, M=args.lots)
    evaluator = Evaluator(market_cfg)

    if args.policy in ("LN", "DR"):
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for learned policies (LN/DR)")
        state = torch.load(args.checkpoint, map_location="cpu")
        # probe state dim via a throwaway env reset
        from rlte.env.execution_env import TradeExecutionEnv
        state_dim = TradeExecutionEnv(market_cfg).reset(seed=0).shape[-1]
        policy = (LogisticNormalPolicy(state_dim, args.K) if args.policy == "LN"
                  else DirichletPolicy(state_dim, args.K))
        policy.load_state_dict(state["policy"])
        policy.eval()
        result = evaluator.evaluate(policy, num_sims=args.num_sims, seed=args.seed,
                                     policy_kind="learned")
    elif args.policy == "SL":
        policy = SubmitAndLeave(K=args.K)
        result = evaluator.evaluate(policy, num_sims=args.num_sims, seed=args.seed,
                                     policy_kind="heuristic")
    elif args.policy == "TWAP":
        policy = TWAP(K=args.K, N=round(market_cfg.T / market_cfg.dt))
        result = evaluator.evaluate(policy, num_sims=args.num_sims, seed=args.seed,
                                     policy_kind="heuristic")
    else:
        raise ValueError(f"Unknown policy '{args.policy}'")

    print(f"{args.policy} | market={args.market} lots={args.lots} | "
          f"E[reward]={result['mean']:.3f} sigma[reward]={result['std']:.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"policy": args.policy, "market": args.market, "lots": args.lots,
                    "mean": result["mean"], "std": result["std"],
                    "num_sims": args.num_sims, "seed": args.seed}, f, indent=2)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
