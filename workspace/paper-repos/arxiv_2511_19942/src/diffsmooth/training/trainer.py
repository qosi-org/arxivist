"""DS-GRPO trainer: a training loop that runs vanilla-GRPO or DS-GRPO depending on config,
sharing every other code path so the comparison between the two is fair (per SIR's need to
isolate the paper's actual contribution from generic RL-training plumbing).

This is a from-scratch loop rather than subclassing trl.GRPOTrainer directly, since our
per-token reward-shaping hook (Eq. 6) needs access to reference log-probs at the advantage
level, which is simplest to wire up explicitly here.
"""
from __future__ import annotations

import torch

from diffsmooth.models.policy import PolicyModel
from diffsmooth.rewards.countdown_reward import CountdownVerifier
from diffsmooth.rewards.differential_smoothing import DifferentialSmoothingShaper
from diffsmooth.training.grpo_advantage import GRPOAdvantage


class DSGRPOTrainer:
    def __init__(self, policy: PolicyModel, verifier: CountdownVerifier, config: dict):
        self.policy = policy
        self.verifier = verifier
        self.advantage_fn = GRPOAdvantage()
        self.shaper = DifferentialSmoothingShaper(
            gamma_p=config["gamma_p"], gamma_n=config["gamma_n"]
        )
        self.use_ds = config.get("use_differential_smoothing", True)
        self.group_size = config["group_size"]
        self.temperature = config["temperature"]
        self.max_new_tokens = config["max_new_tokens"]
        self.clip_low = config["clip_epsilon_low"]
        self.clip_high = config["clip_epsilon_high"]
        self.kl_beta = config["kl_beta"]

        self.optimizer = torch.optim.AdamW(
            self.policy.model.parameters(), lr=config["learning_rate"]
        )

    def train_step(self, batch: list[dict]) -> dict:
        """One GRPO/DS-GRPO update over a batch of Countdown puzzles. Returns loss/metrics dict."""
        prompts = [ex["prompt"] for ex in batch]
        completions = self.policy.generate(
            prompts, self.group_size, self.temperature, self.max_new_tokens
        )  # [B][G]

        rewards = torch.zeros(len(batch), self.group_size)
        ref_logprobs = torch.zeros(len(batch), self.group_size)
        old_logprobs = torch.zeros(len(batch), self.group_size)

        for i, (ex, comps) in enumerate(zip(batch, completions)):
            for g, comp in enumerate(comps):
                rewards[i, g] = self.verifier.score(ex["numbers"], ex["target"], comp)
                lp = self.policy.logprob(ex["prompt"], comp)
                ref_logprobs[i, g] = lp.detach()
                old_logprobs[i, g] = lp

        advantages = self.advantage_fn.compute(rewards)
        is_correct = rewards > 0

        if self.use_ds:
            advantages = self.shaper.shape_advantage(advantages, is_correct, ref_logprobs)

        # Clipped policy-gradient loss (PPO/GRPO-style asymmetric clipping) + forward-KL term (Eq. 1)
        ratio = torch.exp(old_logprobs - old_logprobs.detach())  # == 1 on the sampling step itself;
        # kept for structural parity with a full PPO-style implementation across multiple epochs/step.
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_low, 1 + self.clip_high)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_term = self.kl_beta * old_logprobs.mean()  # ASSUMED simplified KL proxy; see SIR Eq. 1
        loss = policy_loss + kl_term

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "pass_at_1": (rewards[:, 0] > 0).float().mean().item(),
        }
