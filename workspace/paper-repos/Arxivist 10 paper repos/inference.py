#!/usr/bin/env python
"""inference.py -- Run a single trade execution episode with a trained
policy and plot the resulting order book evolution (Figure 1/2/6/7 style).

Usage:
    python inference.py --checkpoint runs/final_checkpoint.pt --market noise --lots 20
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch

from rlte.env.execution_env import MarketConfig, TradeExecutionEnv
from rlte.models.policy import LogisticNormalPolicy
from rlte.utils.config import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one episode and visualize it.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--market", type=str, default="noise")
    parser.add_argument("--lots", type=int, default=20)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--output", type=str, default="results/episode_plot.png")
    args = parser.parse_args()

    set_seed(args.seed)
    market_cfg = MarketConfig(market_type=args.market, K=args.K, M=args.lots)
    env = TradeExecutionEnv(market_cfg)
    state = env.reset(seed=args.seed)
    state_dim = state.shape[-1]

    policy = LogisticNormalPolicy(state_dim, args.K)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    inventories, mids, times = [env.inventory], [env.book.mid_price()], [env.t]
    done = False
    while not done:
        s_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = policy.deterministic_action(s_t).squeeze(0).numpy()
        state, reward, done, info = env.step(a)
        inventories.append(env.inventory)
        mids.append(env.book.mid_price())
        times.append(env.t)
        print(f"t={env.t:6.1f}s | inventory={env.inventory:3d} | mid={env.book.mid_price():.2f} "
              f"| reward={reward:+.4f}")

    if args.plot:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(times, mids, marker="o")
        axes[0].set_ylabel("Mid price")
        axes[1].plot(times, inventories, marker="o", color="tab:green")
        axes[1].set_ylabel("Inventory (lots)")
        axes[1].set_xlabel("Time (s)")
        fig.suptitle(f"Episode: market={args.market}, lots={args.lots}")
        fig.tight_layout()
        fig.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
