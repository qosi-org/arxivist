"""Actor-critic trainer implementing Algorithm 1 of the paper.

Paper reference: Section 4.4, Algorithm 1, Appendix B.1 (Table 5/6
hyperparameters).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.optim import Adam

from rlte.env.execution_env import MarketConfig, VectorizedTradeExecutionEnv
from rlte.models.distributions import LogisticNormalTransform
from rlte.models.policy import LogisticNormalPolicy
from rlte.models.value import ValueNetwork
from rlte.training.losses import LossFunctions


@dataclass
class TrainConfig:
    num_envs: int = 128
    traj_per_env: int = 10
    N: int = 10
    H: int = 400
    learning_rate: float = 5e-4
    adam_beta1: float = 0.9   # ASSUMED (PyTorch default), confidence 0.6
    adam_beta2: float = 0.999  # ASSUMED (PyTorch default), confidence 0.6
    sigma_init: float = 1.0
    sigma_final: float = 0.1
    K: int = 6
    log_every: int = 10
    checkpoint_every: int = 50


class ActorCriticTrainer:
    """Trains a LogisticNormalPolicy + ValueNetwork pair via Algorithm 1."""

    def __init__(self, market_cfg: MarketConfig, train_cfg: TrainConfig, device: str = "cpu"):
        self.market_cfg = market_cfg
        self.cfg = train_cfg
        self.device = torch.device(device)

        probe_env = VectorizedTradeExecutionEnv(market_cfg, num_envs=1)
        state_dim = probe_env.reset(seeds=[0]).shape[-1]

        self.policy = LogisticNormalPolicy(state_dim, train_cfg.K).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.policy_opt = Adam(self.policy.parameters(), lr=train_cfg.learning_rate,
                                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2))
        self.value_opt = Adam(self.value.parameters(), lr=train_cfg.learning_rate,
                               betas=(train_cfg.adam_beta1, train_cfg.adam_beta2))
        self.transform = LogisticNormalTransform()
        self.env = VectorizedTradeExecutionEnv(market_cfg, num_envs=train_cfg.num_envs)
        self.history: list[float] = []

    def summary(self) -> str:
        n_policy = sum(p.numel() for p in self.policy.parameters())
        n_value = sum(p.numel() for p in self.value.parameters())
        return (f"ActorCriticTrainer | policy params={n_policy} value params={n_value} "
                f"| envs={self.cfg.num_envs} traj/env={self.cfg.traj_per_env} N={self.cfg.N} "
                f"| H={self.cfg.H} iterations")

    def collect_trajectories(self, sigma: float) -> dict:
        """Collect `traj_per_env` trajectories from each of `num_envs`
        parallel environments (tau = num_envs * traj_per_env total)."""
        all_states, all_x, all_mu, all_rewards = [], [], [], []
        for _ in range(self.cfg.traj_per_env):
            states = self.env.reset()
            traj_states, traj_x, traj_mu, traj_r = [], [], [], []
            for _n in range(self.cfg.N):
                s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    a, x, mu = self.policy.sample(s_t, sigma)
                actions = a.cpu().numpy()
                next_states, rewards, dones, _infos = self.env.step(actions)
                traj_states.append(states)
                traj_x.append(x.cpu().numpy())
                traj_mu.append(mu.cpu().numpy())
                traj_r.append(rewards)
                states = next_states
            all_states.append(np.stack(traj_states, axis=1))  # [num_envs, N, state_dim]
            all_x.append(np.stack(traj_x, axis=1))
            all_mu.append(np.stack(traj_mu, axis=1))
            all_rewards.append(np.stack(traj_r, axis=1))  # [num_envs, N]

        batch = {
            "states": np.concatenate(all_states, axis=0),   # [tau, N, state_dim]
            "x": np.concatenate(all_x, axis=0),               # [tau, N, K]
            "mu": np.concatenate(all_mu, axis=0),             # [tau, N, K]
            "rewards": np.concatenate(all_rewards, axis=0),   # [tau, N]
        }
        return batch

    def compute_advantages(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eq. 13: A(s_n,a_n) = sum_{l=n}^{N-1} r_l - V(s_n).  GAE lambda=1
        (full Monte Carlo return minus baseline), Remark 4.3."""
        rewards = batch["rewards"]  # [tau, N]
        returns = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1].copy()  # reverse cumsum
        states_flat = torch.as_tensor(batch["states"].reshape(-1, batch["states"].shape[-1]),
                                       dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.value(states_flat).squeeze(-1).cpu().numpy()
        values = values.reshape(rewards.shape)
        advantages = returns - values
        returns_t = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=self.device)
        values_t = torch.as_tensor(values.reshape(-1), dtype=torch.float32, device=self.device)
        return adv_t, returns_t, values_t

    def update_policy(self, batch: dict, advantages: torch.Tensor, sigma: float) -> float:
        states = torch.as_tensor(batch["states"].reshape(-1, batch["states"].shape[-1]),
                                  dtype=torch.float32, device=self.device)
        x = torch.as_tensor(batch["x"].reshape(-1, batch["x"].shape[-1]),
                             dtype=torch.float32, device=self.device)
        mu = self.policy(states)
        log_probs = self.policy.log_prob(mu, x, sigma)
        loss = LossFunctions.policy_loss(advantages, log_probs)
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
        return float(loss.item())

    def update_value(self, batch: dict, returns: torch.Tensor) -> float:
        states = torch.as_tensor(batch["states"].reshape(-1, batch["states"].shape[-1]),
                                  dtype=torch.float32, device=self.device)
        values = self.value(states).squeeze(-1)
        loss = LossFunctions.value_loss(values, returns)
        self.value_opt.zero_grad()
        loss.backward()
        self.value_opt.step()
        return float(loss.item())

    def train(self, num_iterations: int | None = None) -> dict:
        H = num_iterations or self.cfg.H
        print(self.summary())
        for i in range(1, H + 1):
            sigma = LogisticNormalTransform.variance_schedule(
                i, self.cfg.sigma_init, self.cfg.sigma_final, H)
            batch = self.collect_trajectories(sigma)
            advantages, returns, _values = self.compute_advantages(batch)
            p_loss = self.update_policy(batch, advantages, sigma)
            v_loss = self.update_value(batch, returns)
            avg_return = float(batch["rewards"].sum(axis=1).mean())
            self.history.append(avg_return)
            if i % self.cfg.log_every == 0 or i == 1:
                print(f"[iter {i:4d}/{H}] avg_return={avg_return:+.4f} "
                      f"policy_loss={p_loss:+.4f} value_loss={v_loss:.4f} sigma={sigma:.3f}")
        return {"history": self.history}
