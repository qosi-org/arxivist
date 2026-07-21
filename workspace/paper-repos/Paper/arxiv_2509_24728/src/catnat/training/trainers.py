"""
Training loops for all three experiments.

Implements:
  GSLTrainer  — REINFORCE + LOO baseline on Energy Score (Section 5.1, Appendix E)
  VAETrainer  — Gumbel-Softmax ELBO with temperature annealing (Section 5.2, Appendix F)
  PPOTrainer  — Proximal Policy Optimization (Section 5.3, Appendix I.3)

Paper: "Beyond Softmax: A Natural Parameterization for Categorical Random Variables"
arXiv: 2509.24728v2. ICML 2026.
"""

import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..training.losses import EnergyScore, VAELoss
from ..training.baseline import LOOBaseline
from ..samplers import GumbelSoftmaxSampler
from ..utils.config import GSLConfig, VAEConfig, RLConfig


# ---------------------------------------------------------------------------
# GSL Trainer
# ---------------------------------------------------------------------------

class GSLTrainer:
    """Training loop for the Graph Structure Learning experiment.

    Jointly optimises GCN parameters ψ (direct gradient via ES loss) and
    latent graph parameters θ (REINFORCE with LOO baseline).

    Paper: Section 5.1, Appendix E.3–E.4. Confidence: 0.90.
    """

    def __init__(self, model, config: GSLConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = EnergyScore()
        self.baseline_fn = LOOBaseline()
        self._setup_optimizer()

    def _setup_optimizer(self) -> None:
        tc = self.config.training
        # GCN params (ψ) and score params (θ) share the same optimizer and LR
        self.optimizer = Adam(
            self.model.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
    ) -> Dict:
        """Run the full training loop.

        Returns:
            Dict of training history and best validation metric.
        """
        tc = self.config.training
        oc = self.config.output
        Path(oc.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        best_val_es = float("inf")
        history = {"train_es": [], "val_es": []}
        global_step = 0

        # Print training summary
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"GSL Training | parameterization: {self.config.parameterization}")
        print(f"  Trainable params: {n_params:,}")
        print(f"  Dataset size: {len(train_loader.dataset):,} train samples")
        print(f"  Epochs: {tc.epochs} | Batch: {tc.batch_size} | M samples: {tc.M_samples}")
        print(f"  LR: {tc.lr} | Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(tc.epochs):
            self.model.train()
            epoch_es = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                B = x.shape[0]

                # Sample M adjacency matrices — shape [M, N, N]
                with torch.no_grad():
                    A_samples = self.model.sample_graphs(tc.M_samples)  # [M, N, N]

                # Forward: get predictions for each graph sample
                # GSL model broadcasts over batch — reshape for batched GNN
                A_batch = A_samples.unsqueeze(1).expand(
                    tc.M_samples, B, self.config.model.n_nodes, self.config.model.n_nodes
                )  # [M, B, N, N]
                y_preds = self.model(x, A_batch)  # [M, B, N, d_out]

                # Energy Score loss for each sample — [M, B]
                es_per_sample = torch.stack([
                    torch.norm(y_preds[m] - y, dim=-1).mean(dim=-1)
                    for m in range(tc.M_samples)
                ], dim=0)  # [M, B]  (simplified per-sample ES)

                # Full ES loss (Eq. 10) — scalar for ψ gradient
                es_loss = self.loss_fn(y_preds, y)

                # LOO baseline for REINFORCE gradient on θ (Appendix E.3)
                baseline = self.baseline_fn.compute(es_per_sample)   # [M, B]

                # Log-probs of sampled graphs under current θ
                log_p_A = torch.stack([
                    self.model.latent_log_prob(A_samples[[m]])
                    for m in range(tc.M_samples)
                ], dim=0)  # [M, 1] → squeeze
                log_p_A = log_p_A.squeeze(-1).unsqueeze(-1).expand(tc.M_samples, B)  # [M, B]

                # REINFORCE gradient signal (Appendix E.3)
                reinforce_signal = ((es_per_sample - baseline) * log_p_A).mean()

                # Total loss: direct gradient for ψ + REINFORCE for θ
                # We add them so that backward() handles both
                total_loss = es_loss + reinforce_signal

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_es += es_loss.item()
                global_step += 1

                if global_step % tc.log_every_n_steps == 0 and writer:
                    writer.add_scalar("train/es_loss", es_loss.item(), global_step)

            avg_train_es = epoch_es / len(train_loader)
            history["train_es"].append(avg_train_es)

            # Validation
            val_es = self.evaluate(val_loader)
            history["val_es"].append(val_es)

            print(f"Epoch {epoch+1:3d}/{tc.epochs} | "
                  f"Train ES: {avg_train_es:.4f} | Val ES: {val_es:.4f}")

            if writer:
                writer.add_scalar("val/es_loss", val_es, epoch)

            # Save best checkpoint
            if val_es < best_val_es:
                best_val_es = val_es
                ckpt_path = Path(oc.checkpoint_dir) / "best.pt"
                torch.save({"epoch": epoch, "model": self.model.state_dict(),
                            "val_es": val_es}, ckpt_path)

            # Periodic checkpoint
            if (epoch + 1) % tc.save_every_n_epochs == 0:
                ckpt_path = Path(oc.checkpoint_dir) / f"epoch_{epoch+1}.pt"
                torch.save({"epoch": epoch, "model": self.model.state_dict()}, ckpt_path)

        history["best_val_es"] = best_val_es
        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Compute mean Energy Score on a DataLoader."""
        self.model.eval()
        total_es = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            A_samples = self.model.sample_graphs(self.config.training.M_samples)
            B = x.shape[0]
            A_batch = A_samples.unsqueeze(1).expand(
                self.config.training.M_samples, B,
                self.config.model.n_nodes, self.config.model.n_nodes
            )
            y_preds = self.model(x, A_batch)
            total_es += self.loss_fn(y_preds, y).item()
        return total_es / len(loader)


# ---------------------------------------------------------------------------
# VAE Trainer
# ---------------------------------------------------------------------------

class VAETrainer:
    """Training loop for the Categorical VAE experiment.

    Minimises the ELBO with Gumbel-Softmax reparameterization.
    Anneals the Gumbel temperature from τ_init to τ_min during training.

    Paper: Section 5.2, Appendix F. Confidence: 0.92.
    """

    def __init__(self, model, config: VAEConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self._setup_optimizer()
        self._step = 0

    def _setup_optimizer(self) -> None:
        tc = self.config.training
        if tc.optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=tc.lr)
        elif tc.optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=tc.lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer '{tc.optimizer}'")

    def _current_tau(self) -> float:
        """Compute current Gumbel temperature (Appendix F.3)."""
        tc = self.config.training
        tau = tc.gumbel_tau_init * math.exp(-tc.gumbel_tau_anneal_rate * self._step)
        return max(tc.gumbel_tau_min, tau)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        n_epochs: int = 100,
    ) -> Dict:
        """Run the VAE training loop."""
        tc = self.config.training
        oc = self.config.output
        Path(oc.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        best_val_elbo = float("inf")
        history = {"train_elbo": [], "val_elbo": []}

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"VAE Training | parameterization: {self.config.parameterization}")
        print(f"  N={self.config.model.N}, K={self.config.model.K}")
        print(f"  Trainable params: {n_params:,}")
        print(f"  Epochs: {n_epochs} | Batch: {tc.batch_size} | LR: {tc.lr}")
        print(f"{'='*60}\n")

        for epoch in range(n_epochs):
            self.model.train()
            epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0

            for (x, _) in train_loader:  # MNIST: (image, label) — we only need image
                x = x.to(self.device)

                # Update temperature
                tau = self._current_tau()
                self.model.set_temperature(tau)

                out = self.model(x)
                loss_dict = out["elbo_parts"]
                loss = loss_dict["total"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_total += loss_dict["total"].item()
                epoch_recon += loss_dict["recon"].item()
                epoch_kl    += loss_dict["kl"].item()
                self._step  += 1

                if self._step % tc.log_every_n_steps == 0 and writer:
                    writer.add_scalar("train/elbo",  loss_dict["total"].item(), self._step)
                    writer.add_scalar("train/recon", loss_dict["recon"].item(), self._step)
                    writer.add_scalar("train/kl",    loss_dict["kl"].item(),    self._step)
                    writer.add_scalar("train/tau",   tau,                        self._step)

            n_batches = len(train_loader)
            avg = {k: v / n_batches for k, v in
                   [("total", epoch_total), ("recon", epoch_recon), ("kl", epoch_kl)]}
            history["train_elbo"].append(avg["total"])

            val_elbo = self._validate(val_loader)
            history["val_elbo"].append(val_elbo)

            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"ELBO: {avg['total']:.2f} (recon={avg['recon']:.2f}, kl={avg['kl']:.2f}) | "
                  f"Val: {val_elbo:.2f} | τ={tau:.3f}")

            if writer:
                writer.add_scalar("val/elbo", val_elbo, epoch)

            if val_elbo < best_val_elbo:
                best_val_elbo = val_elbo
                ckpt = Path(oc.checkpoint_dir) / "best.pt"
                torch.save({"epoch": epoch, "model": self.model.state_dict(),
                            "val_elbo": val_elbo}, ckpt)

            if (epoch + 1) % tc.save_every_n_epochs == 0:
                ckpt = Path(oc.checkpoint_dir) / f"epoch_{epoch+1}.pt"
                torch.save({"epoch": epoch, "model": self.model.state_dict()}, ckpt)

        history["best_val_elbo"] = best_val_elbo
        return history

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for (x, _) in loader:
            x = x.to(self.device)
            out = self.model(x)
            total += out["elbo_parts"]["total"].item()
        return total / len(loader)


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """PPO training loop for discrete Atari environments.

    Implements Proximal Policy Optimization (Schulman et al. 2017) with
    Generalized Advantage Estimation (GAE, γ=0.99, λ=0.95).

    Paper: Section 5.3, Appendix I.3. Confidence: 0.90.
    Reference implementation: Huang et al. (2022) PPO.
    """

    def __init__(self, agent, config: RLConfig, device: torch.device) -> None:
        self.agent = agent.to(device)
        self.config = config
        self.device = device
        tc = config.training
        self.optimizer = Adam(
            agent.parameters(), lr=tc.lr, eps=tc.adam_epsilon
        )

    def compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: Tensor,
    ) -> Tensor:
        """Compute Generalised Advantage Estimation (GAE).

        Paper: Appendix I.3 — "advantages are calculated using GAE with γ=0.99, λ=0.95."
        Schulman et al. (2015). Confidence: 0.99 (explicitly stated).

        Args:
            rewards:    [T, n_envs]
            values:     [T, n_envs]
            dones:      [T, n_envs]  (1 if episode ended)
            next_value: [n_envs]     (bootstrapped value from last obs)

        Returns:
            advantages: [T, n_envs]
        """
        tc = self.config.training
        T, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(n_envs, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
                next_done = torch.zeros(n_envs, device=self.device)
            else:
                next_val = values[t + 1]
                next_done = dones[t + 1]

            # TD residual: δ = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = (rewards[t]
                     + tc.gamma * next_val * (1.0 - next_done)
                     - values[t])
            # GAE: A_t = δ_t + γλ * (1 - done_{t+1}) * A_{t+1}
            last_adv = delta + tc.gamma * tc.gae_lambda * (1.0 - next_done) * last_adv
            advantages[t] = last_adv

        return advantages

    def train(self, envs: list, writer: Optional[SummaryWriter] = None) -> Dict:
        """Run the PPO training loop.

        Args:
            envs:   List of preprocessed gym environments (length = num_envs).
            writer: Optional TensorBoard SummaryWriter.

        Returns:
            Dict of episodic return history.
        """
        tc = self.config.training
        oc = self.config.output
        Path(oc.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(oc.results_dir).mkdir(parents=True, exist_ok=True)

        n_envs = len(envs)
        num_steps = tc.num_steps
        batch_size = n_envs * num_steps
        minibatch_size = batch_size // tc.num_minibatches

        # Linear LR annealing schedule (Appendix I.5)
        total_updates = tc.total_timesteps // batch_size
        def lr_lambda(update):
            return 1.0 - (update / total_updates)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Rollout buffers
        obs_buf    = torch.zeros(num_steps, n_envs, 4, 84, 84, dtype=torch.uint8)
        act_buf    = torch.zeros(num_steps, n_envs, dtype=torch.long)
        logp_buf   = torch.zeros(num_steps, n_envs)
        rew_buf    = torch.zeros(num_steps, n_envs)
        done_buf   = torch.zeros(num_steps, n_envs)
        val_buf    = torch.zeros(num_steps, n_envs)

        # Reset environments
        obs_list = [env.reset()[0] for env in envs]
        obs = torch.tensor(
            __import__("numpy").stack(obs_list), dtype=torch.uint8, device=self.device
        )
        done = torch.zeros(n_envs, device=self.device)

        global_step = 0
        update = 0
        history = {"episodic_returns": [], "steps": []}

        print(f"\n{'='*60}")
        print(f"PPO Training | parameterization: {self.config.parameterization}")
        print(f"  Total timesteps: {tc.total_timesteps:,} | num_envs: {n_envs}")
        print(f"  Steps per rollout: {num_steps} | Minibatch: {minibatch_size}")
        print(f"{'='*60}\n")

        while global_step < tc.total_timesteps:
            # --- Rollout collection ---
            for step in range(num_steps):
                obs_buf[step] = obs.cpu()
                done_buf[step] = done.cpu()

                with torch.no_grad():
                    out = self.agent.get_action_and_value(obs)
                action = out["action"]
                act_buf[step]  = action.cpu()
                logp_buf[step] = out["log_prob"].cpu()
                val_buf[step]  = out["value"].squeeze(-1).cpu()

                # Step environments
                import numpy as np
                next_obs_list, rewards, terminateds, truncateds, _ = zip(
                    *[env.step(action[i].item()) for i, env in enumerate(envs)]
                )
                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                dones_t   = torch.tensor(
                    [t or tr for t, tr in zip(terminateds, truncateds)],
                    dtype=torch.float32
                )
                rew_buf[step] = rewards_t
                obs = torch.tensor(
                    np.stack(next_obs_list), dtype=torch.uint8, device=self.device
                )
                done = dones_t.to(self.device)
                global_step += n_envs

                # Log episodic returns when episodes end
                for i, (t, tr) in enumerate(zip(terminateds, truncateds)):
                    if t or tr:
                        # Reset that environment
                        new_obs, _ = envs[i].reset()
                        obs[i] = torch.tensor(new_obs, device=self.device)

                if global_step % tc.log_every_n_steps == 0:
                    print(f"  Step {global_step:>8,}/{tc.total_timesteps:,}")

            # --- Compute advantages ---
            with torch.no_grad():
                next_val = self.agent.evaluate_actions(obs, act_buf[-1].to(self.device))[
                    "value"
                ].squeeze(-1).cpu()

            advantages = self.compute_gae(
                rew_buf, val_buf, done_buf, next_val
            )
            returns = advantages + val_buf

            # Normalize advantages per mini-batch (Appendix I.3)
            # Done inside the mini-batch loop below

            # Flatten buffers
            b_obs   = obs_buf.reshape(-1, 4, 84, 84).to(self.device)
            b_act   = act_buf.reshape(-1).to(self.device)
            b_logp  = logp_buf.reshape(-1).to(self.device)
            b_adv   = advantages.reshape(-1).to(self.device)
            b_ret   = returns.reshape(-1).to(self.device)

            # --- PPO update ---
            indices = torch.randperm(batch_size)
            for epoch_ppo in range(tc.update_epochs):
                for start in range(0, batch_size, minibatch_size):
                    mb_idx = indices[start:start + minibatch_size]

                    mb_obs  = b_obs[mb_idx]
                    mb_act  = b_act[mb_idx]
                    mb_logp = b_logp[mb_idx]
                    mb_adv  = b_adv[mb_idx]
                    mb_ret  = b_ret[mb_idx]

                    # Normalize advantages within minibatch (Appendix I.3)
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    eval_out = self.agent.evaluate_actions(mb_obs, mb_act)
                    new_logp = eval_out["log_prob"]
                    entropy  = eval_out["entropy"]
                    new_val  = eval_out["value"].squeeze(-1)

                    # PPO clipped surrogate (Appendix I.3, Eq. 12 in the SIR)
                    ratio = (new_logp - mb_logp).exp()
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * ratio.clamp(1 - tc.clip_coef, 1 + tc.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    vf_loss = 0.5 * (new_val - mb_ret).pow(2).mean()

                    # Entropy bonus
                    ent_loss = entropy.mean()

                    # Total PPO loss (Appendix I.3)
                    loss = pg_loss + tc.vf_coef * vf_loss - tc.ent_coef * ent_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), tc.max_grad_norm)
                    self.optimizer.step()

            scheduler.step()
            update += 1

            if writer:
                writer.add_scalar("train/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("train/value_loss",  vf_loss.item(), global_step)
                writer.add_scalar("train/entropy",     ent_loss.item(), global_step)

            if global_step % tc.save_every_n_steps == 0:
                ckpt = Path(oc.checkpoint_dir) / f"step_{global_step}.pt"
                torch.save({"step": global_step, "agent": self.agent.state_dict()}, ckpt)

        # Save final checkpoint
        torch.save(
            {"step": global_step, "agent": self.agent.state_dict()},
            Path(oc.checkpoint_dir) / "final.pt"
        )
        return history
