"""
inference.py — Single-step inference demo.

Loads a trained model and runs it interactively on one observation,
returning the Q-values and selected action with explanation.

Usage:
    python inference.py --config configs/config.yaml --checkpoint checkpoints/best.pt

Paper: arXiv:2301.08688
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

from apex_lob_trader.utils.config import load_config, set_seeds, get_device
from apex_lob_trader.models.q_network import DuellingQNetwork

ACTION_NAMES = {
    0: "SELL @ bid  (passive sell)",
    1: "SELL @ mid",
    2: "SELL @ ask  (aggressive sell)",
    3: "BUY  @ bid  (aggressive buy)",
    4: "BUY  @ mid",
    5: "BUY  @ ask  (passive buy)",
    6: "SKIP",
}


def parse_args():
    parser = argparse.ArgumentParser(description="APEX DQN single-step inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    device = get_device(cfg)

    env_cfg = cfg["env"]
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

    print(f"\nModel loaded from {args.checkpoint}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nGenerating random observation for demo inference...")

    obs_dim = env_cfg["history_len"] * env_cfg["state_dim_per_step"]
    obs = np.random.randn(obs_dim).astype(np.float32)

    obs_t = torch.FloatTensor(obs).view(
        1, env_cfg["history_len"], env_cfg["state_dim_per_step"]
    ).to(device)

    with torch.no_grad():
        q_values, _ = model(obs_t)

    q_np = q_values.squeeze(0).cpu().numpy()
    best_action = int(np.argmax(q_np))

    print("\nQ-values per action:")
    for i, (name, q) in enumerate(zip(ACTION_NAMES.values(), q_np)):
        marker = " ← SELECTED" if i == best_action else ""
        print(f"  [{i}] {name:<35} Q={q:+.4f}{marker}")

    print(f"\nSelected action: {best_action} — {ACTION_NAMES[best_action]}")


if __name__ == "__main__":
    main()
