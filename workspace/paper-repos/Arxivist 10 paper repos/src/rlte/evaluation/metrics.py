"""Evaluation loop: run a policy over `num_sims` Monte Carlo market
simulations and report expected normalized reward + std dev (Table 1,
Section 6.3).
"""
from __future__ import annotations

import numpy as np
import torch

from rlte.env.execution_env import MarketConfig, TradeExecutionEnv


class Evaluator:
    def __init__(self, market_cfg: MarketConfig):
        self.market_cfg = market_cfg

    def evaluate(self, policy, num_sims: int = 10000, seed: int = 1,
                 policy_kind: str = "learned") -> dict:
        """Args:
            policy: either a torch policy module (LogisticNormalPolicy /
                DirichletPolicy, exposing `deterministic_action`) or a
                heuristic policy exposing `act(state, step)`.
            policy_kind: 'learned' or 'heuristic'.
        """
        env = TradeExecutionEnv(self.market_cfg)
        rewards = np.zeros(num_sims)
        rng = np.random.default_rng(seed)
        for i in range(num_sims):
            ep_seed = int(rng.integers(0, 2**31 - 1))
            state = env.reset(seed=ep_seed)
            done = False
            total_reward = 0.0
            step = 0
            while not done:
                if policy_kind == "learned":
                    s_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        a = policy.deterministic_action(s_t).squeeze(0).numpy()
                else:
                    a = policy.act({"t": step}, step)
                state, r, done, _info = env.step(a)
                total_reward += r
                step += 1
            rewards[i] = total_reward
        return {"mean": float(rewards.mean()), "std": float(rewards.std()), "rewards": rewards}
