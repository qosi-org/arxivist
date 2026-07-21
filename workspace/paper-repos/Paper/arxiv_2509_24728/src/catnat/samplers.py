"""
Gradient estimators for discrete categorical variables.

Implements the two estimators used across the paper's experiments:

1. GumbelSoftmaxSampler — for VAE experiment (Section 5.2, Appendix F.3)
   Jang et al. (2017) / Maddison et al. (2017) Gumbel-Softmax trick.
   Uses Straight-Through estimator for hard discrete samples.

   WARNING (RISK-02, confidence: 0.65): Gumbel noise is added to LEAF LOG-PROBS
   (our assumption). Alternative: inject at node logits. Config flag available.
   TODO: verify against https://github.com/allemanenti/catnat-torch

2. REINFORCESampler — for GSL experiment (Section 5.1, Appendix E.3)
   Score Function gradient estimator (Williams 1992) with Leave-One-Out baseline.

Paper: Appendix E.3 (REINFORCE), Appendix F.3 (Gumbel-Softmax).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GumbelSoftmaxSampler(nn.Module):
    """Gumbel-Softmax reparameterization with Straight-Through estimator.

    Forward pass:
      1. Add Gumbel noise to log-probabilities of leaf nodes (WARNING: RISK-02)
      2. Apply softmax with temperature τ → soft relaxation C_soft
      3. Take argmax and one-hot encode → C_hard
      4. Straight-through: return C_hard + (C_soft - C_soft.detach())
         so gradients flow through C_soft but forward pass sees C_hard.

    Temperature τ is annealed from τ_init to τ_min during training (Appendix F.3).

    Paper: Appendix F.3. Jang et al. (2017).
    """

    def forward(
        self,
        log_probs: Tensor,
        tau: float = 1.0,
        hard: bool = True,
        injection_point: str = "leaf_logprobs",
    ) -> Tensor:
        """Sample using Gumbel-Softmax with optional Straight-Through.

        # WARNING: RISK-02 — injection_point="leaf_logprobs" is assumed.
        # Alternative: injection_point="node_logits" (not implemented here; inject
        # Gumbel noise upstream in CatNat before the tree traversal).
        # TODO: verify against catnat-torch reference implementation.

        Args:
            log_probs:        Log-probabilities from catnat.log_prob(), shape [..., K].
            tau:              Gumbel-Softmax temperature. Lower → closer to one-hot.
            hard:             If True, use Straight-Through (hard one-hot in forward).
            injection_point:  Where Gumbel noise is added. Only "leaf_logprobs" implemented.

        Returns:
            Sampled tensor, shape [..., K].
            If hard=True: one-hot in forward, dense gradients in backward.
            If hard=False: soft relaxation, fully differentiable.
        """
        if injection_point != "leaf_logprobs":
            raise NotImplementedError(
                "injection_point='node_logits' is not yet implemented. "
                "See RISK-02 in architecture_plan.json. "
                "Set injection_point='leaf_logprobs' (default assumption)."
            )

        assert log_probs.shape[-1] >= 2, (
            f"Expected log_probs with at least 2 categories, got shape {log_probs.shape}"
        )

        # Add Gumbel noise to leaf log-probabilities (standard Gumbel-Softmax, Jang et al. 2017)
        # G ~ Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
        gumbels = -torch.empty_like(log_probs).exponential_().log()  # Gumbel(0,1) noise
        y = (log_probs + gumbels) / tau                              # perturbed & scaled

        # Soft sample
        C_soft = F.softmax(y, dim=-1)   # [..., K]

        if not hard:
            return C_soft

        # Hard one-hot via argmax (Straight-Through estimator)
        # Forward: discrete one-hot; backward: gradient flows through C_soft
        k = C_soft.argmax(dim=-1, keepdim=True)
        C_hard = torch.zeros_like(C_soft).scatter_(-1, k, 1.0)
        # Straight-through: C_hard + (C_soft - C_soft.detach())
        return C_hard + (C_soft - C_soft.detach())

    def anneal_temperature(
        self,
        step: int,
        tau_init: float = 1.0,
        tau_min: float = 0.5,
        anneal_rate: float = 3e-5,
    ) -> float:
        """Compute current temperature via exponential annealing.

        tau(t) = max(tau_min, tau_init * exp(-anneal_rate * t))

        Paper: Appendix F.3 — "τ is annealed from 1 to 0.5 using exponential decay rate 3×10⁻⁵"
        Confidence: 1.0 (explicitly stated).

        Args:
            step:        Current training step.
            tau_init:    Initial temperature. Default 1.0.
            tau_min:     Minimum temperature. Default 0.5.
            anneal_rate: Exponential decay rate. Default 3e-5.

        Returns:
            Current temperature float.
        """
        import math
        tau = tau_init * math.exp(-anneal_rate * step)
        return max(tau_min, tau)

    def __repr__(self) -> str:
        return "GumbelSoftmaxSampler()"


class REINFORCESampler(nn.Module):
    """REINFORCE (Score Function) gradient estimator with Leave-One-Out baseline.

    Used for the GSL experiment where the latent graph structure is sampled
    from a Bernoulli distribution parameterized by catnat (K=2).

    The LOO baseline reduces variance without introducing bias:
        b_m = (1/(M-1)) * Σ_{n≠m} L(A_n)

    Gradient estimate (Appendix E.3, Eq. 9 in the paper prose):
        ∇_θ E[L(A)] ≈ (1/M) Σ_m (L(A_m) - b_m) * ∇_θ log P_θ(A_m)

    Paper: Appendix E.3. Williams (1992).
    """

    def sample(self, probs: Tensor, n_samples: int) -> Tensor:
        """Draw n_samples one-hot samples from Cat(probs).

        Args:
            probs:     Categorical probabilities, shape [B, K] or [N_edges, K].
            n_samples: Number of samples M to draw.

        Returns:
            One-hot samples, shape [n_samples, *probs.shape].
        """
        assert probs.dim() >= 1, f"Expected probs with ≥1 dim, got {probs.shape}"
        # Sample indices from categorical distribution
        # probs: [..., K], expand to [M, ..., K]
        expand_shape = (n_samples,) + probs.shape
        probs_expanded = probs.unsqueeze(0).expand(expand_shape)  # [M, ..., K]
        # Flatten batch dims, sample, reshape back
        flat = probs_expanded.reshape(-1, probs.shape[-1])        # [M*..., K]
        indices = torch.multinomial(flat, num_samples=1).squeeze(-1)  # [M*...]
        K = probs.shape[-1]
        one_hot = F.one_hot(indices, num_classes=K).float()       # [M*..., K]
        return one_hot.reshape(expand_shape)                       # [M, ..., K]

    def log_prob(self, samples: Tensor, probs: Tensor) -> Tensor:
        """Compute log probability of one-hot samples under Cat(probs).

        log P(C=k | probs) = Σ_k C_k * log(probs_k)

        Args:
            samples: One-hot samples, shape [M, ..., K].
            probs:   Categorical probabilities, shape [..., K].

        Returns:
            Log-probabilities, shape [M, ...].
        """
        eps = 1e-8
        log_p = torch.log(probs.clamp(min=eps))  # [..., K]
        # samples: [M, ..., K], log_p: [..., K] → broadcast
        return (samples * log_p.unsqueeze(0)).sum(dim=-1)  # [M, ...]

    def __repr__(self) -> str:
        return "REINFORCESampler()"
