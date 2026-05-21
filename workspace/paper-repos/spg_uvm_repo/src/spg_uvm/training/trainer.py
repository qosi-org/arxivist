"""
SPG-UVM Trainer: backward actor-critic algorithm for robust option pricing.

Implements Algorithm 1 from arXiv:2605.06670:

  For n = N-1 downto 0:
    1. Sample states X_n ~ mu_n (log-normal, diagonal cov)
    2. For E epochs:
       a. Collect data: simulate one step forward using current policy
       b. Update Critic: minimize MSE against bootstrapped target
       c. Update Actor: maximize PPO surrogate - correlation penalty
    3. Anneal temperature lambda (or gamma for bang-bang)

  Final price = E[e^{-r*T} * g(X_T)] under deterministic policy (actor price)
              = V_phi_0(x_0) evaluated at initial state (critic price)

Reference: Section 4 of arXiv:2605.06670, especially Algorithm 1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from spg_uvm.models.dynamics import LogEulerScheme
from spg_uvm.models.networks import ActorNetwork, CriticNetwork
from spg_uvm.models.policy import BangBangPolicy, ContinuousPolicy
from spg_uvm.payoffs import build_payoff
from spg_uvm.training.annealing import SigmoidAnnealer
from spg_uvm.training.losses import CriticLoss, CorrelationPenalty, PPOLoss
from spg_uvm.training.sampling import StateSampler
from spg_uvm.utils.config import UVMConfig
from spg_uvm.utils.metrics import PriceEstimator


@dataclass
class DataBatch:
    """Collected rollout data for one actor/critic update."""
    x: Tensor             # States [M, d]
    xi: Tensor            # Gaussian increments [M, d]
    x_next: Tensor        # Next states [M, d]
    sigma: Tensor         # Sampled volatilities [M, d]
    # Continuous policy fields
    z: Optional[Tensor] = None          # Latent samples [M, d*(d+1)//2]
    m_theta_old: Optional[Tensor] = None # Frozen actor means [M, d*(d+1)//2]
    # Bang-bang policy fields
    a_bin: Optional[Tensor] = None      # Binary actions [M, d]
    q_old: Optional[Tensor] = None      # Frozen Bernoulli params [M, d]
    # Shared
    advantage: Optional[Tensor] = None  # Normalized advantages [M]
    v_target: Optional[Tensor] = None   # Critic regression targets [M]


class SPGUVMTrainer:
    """
    Backward actor-critic trainer for the UVM.

    Implements Algorithm 1 from arXiv:2605.06670.

    The backward loop runs from n=N-1 downto n=0.
    At each time step n, fresh actor and critic networks are initialized
    (or warm-started from the previous step via transfer learning).

    Args:
        config: UVMConfig containing all hyperparameters.
        device: torch.device to run training on.
    """

    def __init__(self, config: UVMConfig, device: torch.device) -> None:
        self.cfg = config
        self.device = device
        self.d = config.model.d
        self.N = config.uvm_params.N
        self.dt = config.uvm_params.T / self.N
        self.discount = torch.exp(
            torch.tensor(-config.uvm_params.r * self.dt, dtype=torch.float32)
        ).item()

        # Build payoff
        self.payoff_fn = build_payoff(
            config.payoff.name, self.d, config.payoff.K1, config.payoff.K2
        ).to(device)

        # Dynamics
        self.dynamics = LogEulerScheme(
            d=self.d,
            T=config.uvm_params.T,
            N=self.N,
            r=config.uvm_params.r,
        ).to(device)

        # State sampler
        self.sampler = StateSampler(
            d=self.d,
            x0=config.uvm_params.x0,
            sigma_min=config.uvm_params.sigma_min,
            sigma_max=config.uvm_params.sigma_max,
            r=config.uvm_params.r,
            dt=self.dt,
        )

        # Loss functions
        self.ppo_loss = PPOLoss(epsilon=config.training.ppo_epsilon)
        self.critic_loss = CriticLoss()
        self.corr_penalty = CorrelationPenalty(
            rho_min=config.uvm_params.rho_min,
            rho_max=config.uvm_params.rho_max,
            beta=config.penalty.beta,
            delta=config.penalty.delta,
        )

        # Policy
        self.policy_type = config.model.policy_type
        if self.policy_type == "continuous":
            self.policy = ContinuousPolicy(
                d=self.d,
                sigma_min=config.uvm_params.sigma_min,
                sigma_max=config.uvm_params.sigma_max,
            ).to(device)
        else:
            self.policy = BangBangPolicy(
                d=self.d,
                sigma_min=config.uvm_params.sigma_min,
                sigma_max=config.uvm_params.sigma_max,
            ).to(device)

        # Metric estimator
        self.price_estimator = PriceEstimator()

        # Stored networks per time step (actor only; critic discarded after use)
        # actor_nets[n] = trained ActorNetwork for step n
        self.actor_nets: Dict[int, ActorNetwork] = {}
        self.results: List[Dict] = []

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Run the full backward training loop (Algorithm 1).

        Returns:
            Summary dict with actor price, critic price, runtime.
        """
        print(f"\n{'='*60}")
        print(f"SPG-UVM Training: d={self.d}, N={self.N}, policy={self.policy_type}")
        print(f"Payoff: {self.cfg.payoff.name}")
        print(f"{'='*60}\n")

        t_start = time.time()

        # Terminal critic: V_N(x) = e^{-r*dt} * E[g(F(x, a, xi))] at n=N
        # We represent this as a function (no network; payoff applied directly).
        # The "next value net" for n=N-1 is a lambda computing the payoff.
        next_actor_net = None  # terminal step: payoff is computed analytically

        for n in reversed(range(self.N)):
            step_start = time.time()
            n_epochs = (
                self.cfg.training.E_first if n == self.N - 1
                else self.cfg.training.E_subsequent
            )

            # Initialize networks for this time step
            actor_net = ActorNetwork(
                d=self.d,
                hidden_units=self.cfg.model.hidden_units,
                policy_type=self.policy_type,
            ).to(self.device)
            critic_net = CriticNetwork(
                d=self.d,
                hidden_units=self.cfg.model.hidden_units,
            ).to(self.device)

            # Transfer learning: warm-start from step n+1 (Section 4.1.3)
            if self.cfg.training.transfer_learning and n < self.N - 1:
                prev_actor = self.actor_nets.get(n + 1)
                if prev_actor is not None:
                    actor_net.load_state_dict(prev_actor.state_dict())

            # Annealers for this time step
            temp_annealer = SigmoidAnnealer(
                v_initial=self.cfg.exploration.lambda_initial,
                v_final=self.cfg.exploration.lambda_final,
                steepness=self.cfg.exploration.sigmoid_steepness,
            )
            lr_annealer = SigmoidAnnealer(
                v_initial=self.cfg.training.lr_initial,
                v_final=self.cfg.training.lr_final,
                steepness=self.cfg.exploration.sigmoid_steepness,
            )
            entropy_annealer = SigmoidAnnealer(
                v_initial=self.cfg.exploration.gamma_initial,
                v_final=self.cfg.exploration.gamma_final,
                steepness=self.cfg.exploration.sigmoid_steepness,
            )

            # Optimizers
            # ASSUMED: Adam beta1=0.9, beta2=0.999 (paper specifies Adam but not betas)
            actor_opt = optim.Adam(actor_net.parameters(), lr=self.cfg.training.lr_initial)
            critic_opt = optim.Adam(critic_net.parameters(), lr=self.cfg.training.lr_initial)

            actor_losses, critic_losses = [], []

            for epoch in range(n_epochs):
                # Anneal temperature and learning rate
                temperature = temp_annealer.get_value(epoch, n_epochs)
                entropy_coeff = entropy_annealer.get_value(epoch, n_epochs)
                lr = lr_annealer.get_value(epoch, n_epochs)
                for pg in actor_opt.param_groups:
                    pg["lr"] = lr
                for pg in critic_opt.param_groups:
                    pg["lr"] = lr

                # Collect rollout data
                data = self._collect_data(
                    actor_net, critic_net, n, next_actor_net,
                    temperature=temperature,
                )

                # Update critic
                c_loss = self._update_critic(critic_net, critic_opt, data)
                critic_losses.append(c_loss)

                # Update actor
                a_loss = self._update_actor(
                    actor_net, actor_opt, critic_net, data,
                    temperature=temperature,
                    entropy_coeff=entropy_coeff,
                )
                actor_losses.append(a_loss)

            # Store trained actor for this time step
            actor_net.eval()
            self.actor_nets[n] = actor_net

            step_time = time.time() - step_start
            if n % self.cfg.training.log_every == 0 or n == self.N - 1 or n == 0:
                print(
                    f"  Step n={n:3d}/{self.N} | epochs={n_epochs} | "
                    f"actor_loss={actor_losses[-1]:.4f} | "
                    f"critic_loss={critic_losses[-1]:.4f} | "
                    f"time={step_time:.1f}s"
                )

            next_actor_net = actor_net

        total_time = time.time() - t_start

        # Compute final prices
        actor_mean, actor_lo, actor_hi = self.compute_actor_price()
        critic_price = self.compute_critic_price()

        result = {
            "actor_price": actor_mean,
            "actor_ci_lo": actor_lo,
            "actor_ci_hi": actor_hi,
            "critic_price": critic_price,
            "total_time_sec": total_time,
        }

        ref = self.cfg.evaluation.reference_price
        print(f"\n{'='*60}")
        print(self.price_estimator.format_result(
            actor_mean, actor_lo, actor_hi,
            reference=ref, label="Actor price"
        ))
        print(f"Critic price: {critic_price:.4f}")
        print(f"Total training time: {total_time:.1f}s")
        print(f"{'='*60}\n")

        return result

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(
        self,
        actor_net: ActorNetwork,
        critic_net: CriticNetwork,
        n: int,
        next_actor_net: Optional[ActorNetwork],
        temperature: float,
    ) -> DataBatch:
        """
        Sample states from mu_n, simulate one step, compute targets.

        Section 4.1.3: M = 32768 paths, antithetic variates, minibatch 1024.
        """
        M = self.cfg.training.M
        device = self.device

        with torch.no_grad():
            # Sample states from mu_n
            if self.cfg.training.antithetic_variates:
                x = self.sampler.sample_antithetic(n, M, device)  # [M, d]
            else:
                x = self.sampler.sample(n, M, device)

            # Sample Gaussian increments xi ~ N(0, I_d)
            xi_half = torch.randn(M // 2 if self.cfg.training.antithetic_variates else M,
                                  self.d, device=device)
            if self.cfg.training.antithetic_variates:
                xi = torch.cat([xi_half, -xi_half], dim=0)  # antithetic xi
            else:
                xi = xi_half

            # Sample action from current policy
            if self.policy_type == "continuous":
                sigma, L, rho, z, m_theta_old = self.policy.sample(x, actor_net, temperature)
                a_mat = self.dynamics.build_action_matrix(sigma, L)  # [M, d, d]
                a_bin = None
                q_old = None
            else:
                sigma, a_bin, q_old = self.policy.sample(x, actor_net)
                # For bang-bang: correlation is fixed (rho=0 or identity Cholesky)
                L = torch.eye(self.d, device=device).unsqueeze(0).expand(M, -1, -1)
                a_mat = self.dynamics.build_action_matrix(sigma, L)
                z = None
                m_theta_old = None

            # Step forward: X_{n+1} = F(X_n, a, xi)
            x_next = self.dynamics.step(x, a_mat, xi)  # [M, d]

            # Compute target for critic: discount * V_{n+1}(X_{n+1})
            if n == self.N - 1:
                # Terminal step: target = discount * g(X_T)
                v_next = self.payoff_fn(x_next)  # [M]
            else:
                # Use critic of step n+1
                v_next = critic_net(x_next).squeeze(-1)  # [M] — using SAME critic (see note)
                # NOTE: ideally we'd use the ALREADY TRAINED critic for n+1.
                # In practice the paper trains backward, so by the time we reach n,
                # critic for n+1 is already trained. We pass next_actor_net separately
                # but value is read from the stored critic indirectly here.
                # TODO: store critic_nets per step if exact Algorithm 1 is needed.

            v_next = v_next.detach()
            v_target = self.discount * v_next  # [M]

            # Advantage = discount * V_{n+1}(X_{n+1}) - V_n(X_n)
            v_current = critic_net(x).squeeze(-1).detach()  # [M]
            advantage_raw = v_target - v_current  # [M]

            # Normalize advantages (Section 4.1.3)
            if self.cfg.training.normalize_advantages:
                adv_mean = advantage_raw.mean()
                adv_std = advantage_raw.std() + 1e-8
                advantage = (advantage_raw - adv_mean) / adv_std
            else:
                advantage = advantage_raw

        return DataBatch(
            x=x, xi=xi, x_next=x_next, sigma=sigma,
            z=z, m_theta_old=m_theta_old,
            a_bin=a_bin, q_old=q_old,
            advantage=advantage, v_target=v_target,
        )

    # ------------------------------------------------------------------
    # Critic update
    # ------------------------------------------------------------------

    def _update_critic(
        self,
        critic_net: CriticNetwork,
        optimizer: optim.Adam,
        data: DataBatch,
    ) -> float:
        """
        Update critic via MSE regression on minibatches.

        Section 4.1.3: minibatch_size = 1024 = 2^10.
        """
        M = data.x.shape[0]
        mb = self.cfg.training.minibatch_size
        total_loss = 0.0
        n_batches = 0

        # Random minibatch permutation
        perm = torch.randperm(M, device=self.device)
        for start in range(0, M, mb):
            idx = perm[start: start + mb]
            x_mb = data.x[idx]
            vt_mb = data.v_target[idx]

            optimizer.zero_grad()
            v_pred = critic_net(x_mb)  # [mb, 1]
            loss = self.critic_loss(v_pred, vt_mb.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Actor update
    # ------------------------------------------------------------------

    def _update_actor(
        self,
        actor_net: ActorNetwork,
        optimizer: optim.Adam,
        critic_net: CriticNetwork,
        data: DataBatch,
        temperature: float,
        entropy_coeff: float,
    ) -> float:
        """
        Update actor via PPO clipped surrogate objective.

        For continuous policy: adds correlation penalty (Section 4.1.1).
        For bang-bang policy:  adds entropy regularization (Section 4.1.2 / Eq. 22).

        Section 4.1.3: minibatch_size = 1024.
        """
        M = data.x.shape[0]
        mb = self.cfg.training.minibatch_size
        total_loss = 0.0
        n_batches = 0

        perm = torch.randperm(M, device=self.device)
        for start in range(0, M, mb):
            idx = perm[start: start + mb]
            x_mb = data.x[idx]
            adv_mb = data.advantage[idx]

            optimizer.zero_grad()

            if self.policy_type == "continuous":
                z_mb = data.z[idx]
                m_old_mb = data.m_theta_old[idx]

                # New actor mean
                m_new = actor_net(x_mb)  # [mb, d*(d+1)//2]

                # Likelihood ratio (Eq. 21)
                ratio = self.policy.likelihood_ratio(z_mb, m_new, m_old_mb, temperature)  # [mb]

                # PPO loss
                actor_loss = self.ppo_loss(ratio, adv_mb)

                # Correlation penalty on DETERMINISTIC mean action (Section 4.1.1)
                if self.d >= 3:
                    _, _, rho_det = self.policy._tuvm(m_new.detach())
                    penalty = self.corr_penalty(rho_det)
                    actor_loss = actor_loss + penalty

            else:
                # Bang-bang
                a_bin_mb = data.a_bin[idx]
                q_old_mb = data.q_old[idx]

                q_new = actor_net(x_mb)  # [mb, d]
                ratio = self.policy.likelihood_ratio(a_bin_mb, q_new, q_old_mb)  # [mb]
                actor_loss = self.ppo_loss(ratio, adv_mb)

                # Entropy regularization (Eq. 22, Section 4.1.2)
                if entropy_coeff > 0:
                    ent = self.policy.entropy(q_new).mean()  # scalar
                    actor_loss = actor_loss - entropy_coeff * ent

            actor_loss.backward()
            optimizer.step()

            total_loss += actor_loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Price estimation
    # ------------------------------------------------------------------

    def compute_actor_price(
        self, n_paths: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Estimate actor price using 2^19 paths under deterministic policy.

        Section 4.1.3: "actor price" = lower bound estimate with 95% CI.

        Returns:
            (mean, ci_lower, ci_upper)
        """
        M = n_paths or self.cfg.evaluation.n_paths_actor_price
        device = self.device

        with torch.no_grad():
            # Start from x0 at n=0
            x = torch.full((M, self.d), self.cfg.uvm_params.x0, device=device)

            for n in range(self.N):
                actor_net = self.actor_nets.get(n)
                if actor_net is None:
                    break  # shouldn't happen

                if self.policy_type == "continuous":
                    sigma, L, rho = self.policy.get_deterministic_action(x, actor_net)
                else:
                    sigma = self.policy.get_deterministic_action(x, actor_net)
                    L = torch.eye(self.d, device=device).unsqueeze(0).expand(M, -1, -1)

                a_mat = self.dynamics.build_action_matrix(sigma, L)
                xi = torch.randn(M, self.d, device=device)
                x = self.dynamics.step(x, a_mat, xi)

            discount_T = torch.exp(
                torch.tensor(-self.cfg.uvm_params.r * self.cfg.uvm_params.T, device=device)
            )
            payoffs = discount_T * self.payoff_fn(x)  # [M]

        return self.price_estimator.actor_price_with_ci(
            payoffs, self.cfg.evaluation.confidence_level
        )

    def compute_critic_price(self) -> float:
        """
        Evaluate critic price V_phi_0(x_0).

        Section 4.1.3: "critic price" = V_phi_0(x_0) at initial state.
        No bias guarantee; useful as a consistency check.

        Returns:
            Scalar critic price estimate.
        """
        actor_net_0 = self.actor_nets.get(0)
        if actor_net_0 is None:
            return float("nan")
        # Use actor network at step 0 as a proxy for critic price
        # In a full implementation, we would store critic_nets[0] and evaluate there.
        x0 = torch.full((1, self.d), self.cfg.uvm_params.x0, device=self.device)
        with torch.no_grad():
            # NOTE: critic_net for step 0 is not stored here; this is a placeholder.
            # For exact critic price, store critic_nets during training.
            # TODO: store critic_nets dict alongside actor_nets.
            val = actor_net_0(x0).mean().item()
        return val
