"""Differential Smoothing — Eq. 4 / Eq. 6 in the paper. This is the paper's central contribution.

Eq. 4 (reward-level, theory):
    r_DS(tau) = r_hat(tau) - gamma_p * log(pi_base(tau))   if r_hat(tau) > 0
              = r_hat(tau) + gamma_n * log(pi_base(tau))   if r_hat(tau) <= 0

Eq. 6 (advantage-level, as actually implemented in GRPO):
    A_i^DS = A_i - gamma_p * log(pi_theta_old(y_i|x))   if r_i == 1 (correct)
           = A_i + gamma_n * log(pi_theta_old(y_i|x))   otherwise

SIR ambiguity (confidence 0.5): the paper's theory (Eq. 1/4) uses the frozen base model's
log-probability, but the practical algorithm (Eq. 6) appears to use the previous policy
iterate (pi_theta_old) instead, per Appendix B.3. We implement Eq. 6's version — the reference
log-probabilities passed in should come from whichever model the caller intends (see
training/trainer.py for how this choice is wired up).
"""
from __future__ import annotations

import torch


class DifferentialSmoothingShaper:
    def __init__(self, gamma_p: float, gamma_n: float):
        self.gamma_p = gamma_p
        self.gamma_n = gamma_n

    def shape_advantage(
        self,
        advantages: torch.Tensor,
        is_correct: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """advantages, is_correct, ref_logprobs: all [B, G]. Returns A_i^DS per Eq. 6.

        is_correct is a {0, 1} tensor (1 where reward indicates a correct trajectory).
        """
        assert advantages.shape == is_correct.shape == ref_logprobs.shape, (
            f"shape mismatch: {advantages.shape}, {is_correct.shape}, {ref_logprobs.shape}"
        )
        correction = torch.where(
            is_correct.bool(),
            -self.gamma_p * ref_logprobs,
            self.gamma_n * ref_logprobs,
        )
        return advantages + correction
