"""
training/trainer.py — APEX Deep Double Duelling DQN Trainer.

Implements the APEX training architecture (Section 3.2):
  - Asynchronous parallel environment workers (CPU)
  - Single GPU learner with uniform replay buffer
  - Double Q-learning: main network selects actions, target network evaluates
  - Target network soft-updated every target_update_freq steps
  - N-step returns (n_step=3, Table 2)
  - Learning rate schedule: [[0, 2e-5], [1e6, 5e-6]] (Table 2)

Full-scale APEX requires RLlib (see configs/config.yaml: framework=torch,
num_workers=42). This trainer provides a single-process reference
implementation for reproducibility and debugging.

Paper: arXiv:2301.08688 — Section 3.2, Table 2.
References: Horgan et al. (2018) APEX [5]; Van Hasselt et al. (2016) DDQN [26]
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from apex_lob_trader.models.q_network import DuellingQNetwork
from apex_lob_trader.training.replay_buffer import ReplayBuffer, Transition
from apex_lob_trader.training.environment import LOBTradingEnv


class APEXDQNTrainer:
    """Single-process reference implementation of APEX Deep Double Duelling DQN.

    For full distributed training with 42 workers, use the RLlib configuration
    in configs/config.yaml with: python train.py --config configs/config.yaml --use-rllib

    Args:
        env: LOBTradingEnv instance.
        cfg: Full config dictionary.
        device: torch.device for the learner.
    """

    def __init__(
        self,
        env: LOBTradingEnv,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.device = device

        model_cfg = cfg["model"]
        train_cfg = cfg["training"]
        env_cfg = cfg["env"]

        obs_dim = env_cfg["history_len"] * env_cfg["state_dim_per_step"]

        # ── Networks (main + target for Double Q-learning) ─────────────────
        # Section 3.2: Double DQN keeps separate weights for selection and validation
        self.main_net = DuellingQNetwork(
            state_dim=env_cfg["state_dim_per_step"],
            history_len=env_cfg["history_len"],
            hidden_dim=model_cfg["hidden_dim"],
            num_actions=model_cfg["num_actions"],
            num_ff_layers=model_cfg["num_ff_layers"],
        ).to(device)

        self.target_net = DuellingQNetwork(
            state_dim=env_cfg["state_dim_per_step"],
            history_len=env_cfg["history_len"],
            hidden_dim=model_cfg["hidden_dim"],
            num_actions=model_cfg["num_actions"],
            num_ff_layers=model_cfg["num_ff_layers"],
        ).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        # ── Replay buffer ──────────────────────────────────────────────────
        learn_cfg = train_cfg["learning"]
        self.buffer = ReplayBuffer(
            capacity=int(train_cfg["replay_buffer"]["size"]),
            obs_dim=obs_dim,
            seed=cfg.get("seed", 42),
        )
        self.batch_size: int = learn_cfg["train_batch_size"]
        self.learning_starts: int = learn_cfg["learning_starts"]
        self.target_update_freq: int = learn_cfg["target_update_freq"]
        self.n_step: int = learn_cfg["n_step"]
        self.gamma: float = train_cfg["gamma"]

        # ── Optimizer with linear LR schedule (Table 2) ────────────────────
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=2e-5)
        lr_schedule = train_cfg["lr_schedule"]
        self._lr_schedule = [(int(s[0]), float(s[1])) for s in lr_schedule]

        # ── Training state ─────────────────────────────────────────────────
        self.total_steps: int = 0
        self.total_timesteps: int = int(train_cfg["total_timesteps"])
        self.log_every: int = cfg.get("logging", {}).get("log_every_n_steps", 1000)
        self.ckpt_every: int = cfg.get("logging", {}).get("checkpoint_every_n_steps", 50000)
        self.ckpt_dir = Path(cfg.get("logging", {}).get("checkpoint_dir", "checkpoints/"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.episode_returns: list[float] = []
        self.losses: list[float] = []

    def _get_lr(self, step: int) -> float:
        """Interpolate learning rate from schedule (Table 2)."""
        schedule = self._lr_schedule
        if step <= schedule[0][0]:
            return schedule[0][1]
        if step >= schedule[-1][0]:
            return schedule[-1][1]
        for i in range(len(schedule) - 1):
            s0, lr0 = schedule[i]
            s1, lr1 = schedule[i + 1]
            if s0 <= step < s1:
                frac = (step - s0) / (s1 - s0)
                return lr0 + frac * (lr1 - lr0)
        return schedule[-1][1]

    def _update_lr(self) -> None:
        """Update optimizer learning rate according to schedule."""
        lr = self._get_lr(self.total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.env.action_space.n))
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        obs_t = obs_t.view(1, self.cfg["env"]["history_len"], self.cfg["env"]["state_dim_per_step"])
        obs_t = obs_t.to(self.device)
        with torch.no_grad():
            q_vals, _ = self.main_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def _learn(self) -> float:
        """Single gradient update step (Double DQN loss, Section 3.2).

        Returns:
            Loss value (float).
        """
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        h = self.cfg["env"]["history_len"]
        d = self.cfg["env"]["state_dim_per_step"]

        states_t = torch.FloatTensor(states).view(-1, h, d).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).view(-1, h, d).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        q_values, _ = self.main_net(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: main net selects action, target net evaluates (Section 3.2, [26])
        with torch.no_grad():
            next_q_main, _ = self.main_net(next_states_t)
            next_actions = next_q_main.argmax(dim=1)
            next_q_target, _ = self.target_net(next_states_t)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.gamma ** self.n_step * next_q * (~dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    def train(self, debug: bool = False) -> None:
        """Run the training loop.

        Args:
            debug: If True, run a short loop (200 steps) for quick validation.
        """
        max_steps = 2000 if debug else self.total_timesteps
        epsilon_start, epsilon_end = 1.0, 0.01
        epsilon_decay = 0.995

        print(f"\n{'='*60}")
        print(f"  APEX LOB Trader — Training")
        print(f"  Model params: {sum(p.numel() for p in self.main_net.parameters()):,}")
        print(f"  Replay buffer: {self.buffer.capacity:,}")
        print(f"  Max steps: {max_steps:,}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_steps = 0
        epsilon = epsilon_start
        t_start = time.time()

        for step in range(max_steps):
            self.total_steps = step

            # Act
            action = self._select_action(obs, epsilon)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.push(Transition(obs, action, reward, next_obs, done))
            obs = next_obs
            episode_return += reward
            episode_steps += 1

            if done:
                self.episode_returns.append(episode_return)
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_steps = 0
                epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Learn
            if step >= self.learning_starts and len(self.buffer) >= self.batch_size:
                loss = self._learn()
                self.losses.append(loss)
                self._update_lr()

            # Target network update (Section 3.2, Table 2)
            if step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())

            # Logging
            if step % self.log_every == 0 and step > 0:
                elapsed = time.time() - t_start
                avg_return = np.mean(self.episode_returns[-20:]) if self.episode_returns else 0.0
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
                lr = self._get_lr(step)
                print(
                    f"Step {step:>10,} | "
                    f"Eps {epsilon:.3f} | "
                    f"AvgReturn {avg_return:.4f} | "
                    f"Loss {avg_loss:.5f} | "
                    f"LR {lr:.2e} | "
                    f"Buf {len(self.buffer):,} | "
                    f"Elapsed {elapsed:.0f}s"
                )

            # Checkpointing
            if step % self.ckpt_every == 0 and step > 0:
                self._save_checkpoint(step)

        self._save_checkpoint(max_steps, tag="final")
        print(f"\nTraining complete. {max_steps:,} steps in {time.time()-t_start:.0f}s")

    def _save_checkpoint(self, step: int, tag: str = "") -> None:
        """Save model checkpoint."""
        fname = f"checkpoint_step{step}{('_'+tag) if tag else ''}.pt"
        path = self.ckpt_dir / fname
        torch.save({
            "step": step,
            "main_net_state_dict": self.main_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_returns": self.episode_returns,
        }, path)
        print(f"  [Checkpoint] Saved → {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and return step number.

        Args:
            path: Path to .pt checkpoint file.

        Returns:
            Step number when checkpoint was saved.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.main_net.load_state_dict(ckpt["main_net_state_dict"])
        self.target_net.load_state_dict(ckpt["target_net_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.episode_returns = ckpt.get("episode_returns", [])
        step = ckpt.get("step", 0)
        print(f"[Checkpoint] Loaded from {path} (step {step:,})")
        return step
