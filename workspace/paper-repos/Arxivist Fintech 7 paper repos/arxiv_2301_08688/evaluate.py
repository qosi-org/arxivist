"""
evaluate.py — Evaluation Entrypoint.

Evaluates a trained RL policy against the heuristic baseline and buy-and-hold
on the test set. Reports mean log-return, Sharpe ratio, and outperformance.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --noise-level 1.1

Paper: arXiv:2301.08688 — Section 5.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

from apex_lob_trader.utils.config import load_config, set_seeds, get_device
from apex_lob_trader.data.lob_dataset import LOBDataset
from apex_lob_trader.training.environment import LOBTradingEnv
from apex_lob_trader.models.q_network import DuellingQNetwork
from apex_lob_trader.evaluation.baseline import HeuristicBaseline
from apex_lob_trader.evaluation.metrics import (
    mean_log_return, sharpe_ratio, outperformance_pp,
    bootstrap_ci, print_results_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained APEX DQN policy (arXiv:2301.08688)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--noise-level", type=float, default=None,
                        help="Override signal noise level (a_H): 1.1, 1.3, or 1.6.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def run_episode(env: LOBTradingEnv, policy_fn, greedy: bool = True) -> dict:
    """Run one evaluation episode and return metrics."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    actions = []
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        actions.append(action)
    return {"return": total_reward, "actions": actions, "final_M": info.get("M", 0.0)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.noise_level is not None:
        cfg["signal"]["noise_level_a"] = args.noise_level

    set_seeds(cfg["seed"])
    device = get_device(cfg)

    # Load test dataset
    env_cfg = cfg["env"]
    dataset = LOBDataset(
        data_dir=env_cfg["data"]["data_dir"],
        asset=env_cfg["data"]["asset"],
        split="test",
        history_len=env_cfg["history_len"],
    )
    dataset.load()
    env = LOBTradingEnv(dataset=dataset, cfg=cfg)

    # Load trained model
    model_cfg = cfg["model"]
    model = DuellingQNetwork(
        state_dim=env_cfg["state_dim_per_step"],
        history_len=env_cfg["history_len"],
        hidden_dim=model_cfg["hidden_dim"],
        num_actions=model_cfg["num_actions"],
        num_ff_layers=model_cfg["num_ff_layers"],
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["main_net_state_dict"])
    model.eval()

    def rl_policy(obs: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).view(
            1, env_cfg["history_len"], env_cfg["state_dim_per_step"]
        ).to(device)
        with torch.no_grad():
            q, _ = model(obs_t)
        return int(q.argmax(dim=1).item())

    baseline = HeuristicBaseline(
        pos_min=env_cfg["inventory"]["pos_min"],
        pos_max=env_cfg["inventory"]["pos_max"],
    )

    def baseline_policy(obs: np.ndarray) -> int:
        # Extract signal from observation (positions 3:6 in each step, first step)
        step_dim = env_cfg["state_dim_per_step"]
        signal = obs[3:6]  # first history step's signal
        inv_idx = 2
        inventory = int(round(obs[inv_idx] * env_cfg["inventory"]["pos_max"]))
        return baseline.act(signal, inventory)

    # Evaluate
    n_episodes = len(dataset)
    print(f"\nEvaluating on {n_episodes} test episodes (noise a={cfg['signal']['noise_level_a']})...")

    rl_returns, baseline_returns = [], []
    for i in range(n_episodes):
        rl_ep = run_episode(env, rl_policy)
        base_ep = run_episode(env, baseline_policy)
        rl_returns.append(rl_ep["return"])
        baseline_returns.append(base_ep["return"])
        if (i + 1) % 5 == 0:
            print(f"  Episode {i+1}/{n_episodes} done.")

    # Metrics
    rl_ci = bootstrap_ci(rl_returns)
    base_ci = bootstrap_ci(baseline_returns)

    results = {
        "RL (APEX DQN)": {
            "mean_return": mean_log_return(rl_returns),
            "sharpe": sharpe_ratio(rl_returns),
            "turnover": float("nan"),  # requires action log
        },
        "Heuristic Baseline": {
            "mean_return": mean_log_return(baseline_returns),
            "sharpe": sharpe_ratio(baseline_returns),
            "turnover": float("nan"),
        },
    }

    print_results_table(results)
    print(f"RL 95% CI: ({rl_ci[0]:.4f}, {rl_ci[1]:.4f})")
    print(f"Baseline 95% CI: ({base_ci[0]:.4f}, {base_ci[1]:.4f})")
    print(f"Outperformance: {outperformance_pp(rl_returns, baseline_returns):.1f} pp")

    # Paper expected results (Table from Section 5)
    print("\nPaper reported results for comparison:")
    paper = {
        1.1: {"rl_mu": 0.00, "rl_S": -0.72, "outperf_pp": 32.2},
        1.3: {"rl_mu": 0.11, "rl_S":  7.34, "outperf_pp": 14.8},
        1.6: {"rl_mu": 0.21, "rl_S": 14.69, "outperf_pp": 20.7},
    }
    a = cfg["signal"]["noise_level_a"]
    if a in paper:
        p = paper[a]
        print(f"  Noise a={a}: μ={p['rl_mu']}, S={p['rl_S']}, outperf={p['outperf_pp']}pp")


if __name__ == "__main__":
    main()
