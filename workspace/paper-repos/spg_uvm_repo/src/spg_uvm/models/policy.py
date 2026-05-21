"""
Policy classes for SPG-UVM.

Implements two policy families from Section 3.2 of arXiv:2605.06670:

1. ContinuousPolicy: Squashed Gaussian on (sigma, rho) via C-vine parameterization.
   - Supports uncertain volatility AND uncertain correlations.
   - Key property: positive semidefiniteness enforced by construction.

2. BangBangPolicy: Factorized Bernoulli on {sigma_min, sigma_max}^d.
   - For uncertain volatility only (correlations handled separately or fixed).
   - Theoretically exact in d=1 (optimal control is bang-bang, Eq. (5)).
   - Scales linearly in d (not exponentially).

Both policies are stochastic during training and are annealed toward
deterministic policies at inference time (Section 4.1.2).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from spg_uvm.models.networks import ActorNetwork
from spg_uvm.models.vine import CVineCorrelation


# ---------------------------------------------------------------------------
# Continuous (Squashed Gaussian) Policy
# ---------------------------------------------------------------------------

class ContinuousPolicy(nn.Module):
    """
    Squashed Gaussian policy over (sigma, rho) for the UVM.

    The latent variable z ~ N(m_theta(x), lambda * I) is squashed via TUVM:
      - Volatility: sigma^i = sigma_mid^i + sigma_half^i * tanh(z^i_sigma)
      - Correlation: (z_rho components) -> C-vine -> PSD correlation matrix rho

    Section 3.2.1, arXiv:2605.06670.

    Args:
        d:         Number of assets.
        sigma_min: List[float], length d.
        sigma_max: List[float], length d.
        eps:       Clamp margin for tanh outputs.
    """

    def __init__(
        self,
        d: int,
        sigma_min: list,
        sigma_max: list,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_sigma = d
        self.n_rho = d * (d - 1) // 2
        self.n_latent = d * (d + 1) // 2  # = n_sigma + n_rho

        # Register volatility bounds as buffers (non-trainable)
        sigma_min_t = torch.tensor(sigma_min, dtype=torch.float32)
        sigma_max_t = torch.tensor(sigma_max, dtype=torch.float32)
        self.register_buffer("sigma_min", sigma_min_t)
        self.register_buffer("sigma_max", sigma_max_t)
        self.register_buffer("sigma_mid", (sigma_min_t + sigma_max_t) / 2.0)
        self.register_buffer("sigma_half", (sigma_max_t - sigma_min_t) / 2.0)

        self.eps = eps
        self.vine = CVineCorrelation(d, eps=eps)

    def _tuvm(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply the TUVM map to latent z.

        z = (z_sigma [B,d], z_rho [B, d*(d-1)//2])

        Returns:
            sigma: [B, d] — admissible volatilities
            L:     [B, d, d] — Cholesky factor of correlation matrix
            rho:   [B, d, d] — correlation matrix
        """
        # tanh squashing of entire latent vector (Section 3.2.1)
        tanh_z = torch.tanh(z).clamp(-1 + self.eps, 1 - self.eps)

        z_sigma = tanh_z[:, : self.n_sigma]  # [B, d]
        z_rho = tanh_z[:, self.n_sigma :]    # [B, d*(d-1)//2]

        # Volatility: affine rescaling from (-1,1) to (sigma_min, sigma_max)
        # Eq. in Section 3.2.1: sigma^i = sigma_mid + sigma_half * tanh(z^i_sigma)
        sigma = self.sigma_mid + self.sigma_half * z_sigma  # [B, d]

        # Correlation: C-vine map from partial correlations to PSD rho
        # Section 3.2.1: C-vine enforces PSD by construction
        if self.d == 1:
            # No correlation component
            L = torch.ones(z.shape[0], 1, 1, device=z.device, dtype=z.dtype)
            rho = torch.ones(z.shape[0], 1, 1, device=z.device, dtype=z.dtype)
        elif self.d == 2:
            # Single correlation; PSD automatic for rho_12 in (-1,1)
            # Remark 3.2: C-vine map reduces to identity for d=2
            rho_12 = z_rho[:, 0:1]  # [B, 1]
            L = torch.zeros(z.shape[0], 2, 2, device=z.device, dtype=z.dtype)
            L[:, 0, 0] = 1.0
            L[:, 1, 0] = rho_12[:, 0]
            L[:, 1, 1] = torch.sqrt((1 - rho_12[:, 0] ** 2).clamp(min=self.eps))
            rho = torch.bmm(L, L.transpose(1, 2))
        else:
            # d >= 3: use C-vine recursion
            L, rho = self.vine(z_rho)  # [B,d,d], [B,d,d]

        return sigma, L, rho

    def sample(
        self,
        x: Tensor,
        actor_net: ActorNetwork,
        temperature: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample an action from the continuous policy.

        a ~ pi_theta(· | x) = TUVM_# N(m_theta(x), lambda * I)

        Args:
            x:           State [B, d].
            actor_net:   ActorNetwork (outputs mean m_theta).
            temperature: Current lambda (std of latent Gaussian).

        Returns:
            sigma:   [B, d] — sampled volatilities
            L:       [B, d, d] — Cholesky factor of rho
            rho:     [B, d, d] — correlation matrix
            z:       [B, d*(d+1)//2] — latent sample (for likelihood ratio computation)
            m_theta: [B, d*(d+1)//2] — actor mean (stored for PPO ratio)
        """
        m_theta = actor_net(x)  # [B, d*(d+1)//2]
        # Reparameterized sample: z ~ N(m_theta, lambda^2 * I)
        eps_noise = torch.randn_like(m_theta)
        z = m_theta + temperature * eps_noise  # [B, d*(d+1)//2]
        sigma, L, rho = self._tuvm(z)
        return sigma, L, rho, z, m_theta

    def get_deterministic_action(
        self, x: Tensor, actor_net: ActorNetwork
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Deterministic action TUVM(m_theta(x)) for inference.

        Section 3.3: "deterministic policy obtained by TUVM(m_theta*(x))"

        Returns:
            sigma, L, rho
        """
        with torch.no_grad():
            m_theta = actor_net(x)
            sigma, L, rho = self._tuvm(m_theta)
        return sigma, L, rho

    def likelihood_ratio(
        self,
        z: Tensor,
        m_theta_new: Tensor,
        m_theta_old: Tensor,
        temperature: float,
    ) -> Tensor:
        """
        Compute PPO likelihood ratio p_theta(a|x) / p_theta_old(a|x).

        Key insight (Section 3.3, Eq. (21)): the Jacobian factors from TUVM cancel,
        so the ratio reduces to a Gaussian ratio in latent space:

            ratio = exp[(m_new - m_old)^T * Lambda^{-1} * (z - 0.5*(m_new+m_old))]

        where Lambda = lambda^2 * I.

        Args:
            z:             Latent samples used to generate actions, [B, d*(d+1)//2].
            m_theta_new:   New actor mean, [B, d*(d+1)//2].
            m_theta_old:   Old actor mean (frozen), [B, d*(d+1)//2].
            temperature:   Current lambda (std).

        Returns:
            ratio: [B], the likelihood ratio.
        """
        # Eq. (21) from arXiv:2605.06670:
        # ratio = exp[(m_new - m_old)^T Lambda^{-1} (z - 0.5(m_new + m_old))]
        lam2 = temperature ** 2
        delta_m = m_theta_new - m_theta_old  # [B, d*(d+1)//2]
        z_shift = z - 0.5 * (m_theta_new + m_theta_old)  # [B, d*(d+1)//2]
        log_ratio = (delta_m * z_shift / lam2).sum(dim=-1)  # [B]
        return torch.exp(log_ratio.clamp(-20.0, 20.0))  # clamp for numerical stability

    def __repr__(self) -> str:
        return f"ContinuousPolicy(d={self.d})"


# ---------------------------------------------------------------------------
# Bang-Bang (Bernoulli) Policy
# ---------------------------------------------------------------------------

class BangBangPolicy(nn.Module):
    """
    Factorized Bernoulli policy for bang-bang volatility control.

    At each time step, the i-th volatility is independently sampled:
        sigma^i = sigma_min^i + a^i * (sigma_max^i - sigma_min^i)
        a^i ~ Bernoulli(q^i_theta(x))

    Section 3.2.2, arXiv:2605.06670.

    Density (Eq. 19): p_theta(a|x) = prod_i (q^i)^{a^i} * (1-q^i)^{1-a^i}

    Scaling advantage: output dimension = d (linear), not 2^d (exponential).

    Args:
        d:         Number of assets.
        sigma_min: List[float], length d.
        sigma_max: List[float], length d.
    """

    def __init__(self, d: int, sigma_min: list, sigma_max: list) -> None:
        super().__init__()
        self.d = d
        sigma_min_t = torch.tensor(sigma_min, dtype=torch.float32)
        sigma_max_t = torch.tensor(sigma_max, dtype=torch.float32)
        self.register_buffer("sigma_min", sigma_min_t)
        self.register_buffer("sigma_max", sigma_max_t)
        self.register_buffer("sigma_range", sigma_max_t - sigma_min_t)

    def _binary_to_sigma(self, a: Tensor) -> Tensor:
        """Map binary a in {0,1}^d to admissible volatility vector.

        sigma^i = sigma_min^i + a^i * (sigma_max^i - sigma_min^i)
        Section 3.2.2 of arXiv:2605.06670.
        """
        return self.sigma_min + a * self.sigma_range  # [B, d]

    def sample(
        self, x: Tensor, actor_net: ActorNetwork
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample bang-bang action from factorized Bernoulli policy.

        Args:
            x:          State [B, d].
            actor_net:  ActorNetwork outputting Bernoulli params q [B, d] in (0,1).

        Returns:
            sigma:  [B, d] — sampled volatilities (values in {sigma_min, sigma_max})
            a_bin:  [B, d] — binary action in {0,1}^d
            q:      [B, d] — Bernoulli parameters (stored for PPO ratio)
        """
        q = actor_net(x)  # [B, d], already sigmoid-activated
        a_bin = torch.bernoulli(q)  # [B, d], in {0.0, 1.0}
        sigma = self._binary_to_sigma(a_bin)
        return sigma, a_bin, q

    def get_deterministic_action(
        self, x: Tensor, actor_net: ActorNetwork
    ) -> Tensor:
        """
        Deterministic bang-bang action: a^i = 1{q^i >= 0.5}.

        Section 3.3: "deterministic policy obtained by replacing each Bernoulli
        parameter q^i by the deterministic decision 1{q^i >= 1/2}."

        Returns:
            sigma: [B, d] — deterministic volatility choices
        """
        with torch.no_grad():
            q = actor_net(x)
            a_bin = (q >= 0.5).float()
        return self._binary_to_sigma(a_bin)

    def log_prob(self, a_bin: Tensor, q: Tensor) -> Tensor:
        """
        Log-density of factorized Bernoulli: sum_i log p(a^i | q^i).

        Eq. (19) from arXiv:2605.06670.

        Args:
            a_bin: Binary actions [B, d] in {0, 1}.
            q:     Bernoulli parameters [B, d] in (0, 1).

        Returns:
            log_prob: [B]
        """
        eps = 1e-8
        q_clamped = q.clamp(eps, 1 - eps)
        log_p = a_bin * torch.log(q_clamped) + (1 - a_bin) * torch.log(1 - q_clamped)
        return log_p.sum(dim=-1)  # [B]

    def likelihood_ratio(
        self, a_bin: Tensor, q_new: Tensor, q_old: Tensor
    ) -> Tensor:
        """
        PPO likelihood ratio for bang-bang policy.

        Section 3.3 (after Eq. (21)):
            ratio = prod_i (q_new^i / q_old^i)^{a^i} * ((1-q_new^i)/(1-q_old^i))^{1-a^i}

        Args:
            a_bin:  Binary action [B, d].
            q_new:  New Bernoulli params [B, d].
            q_old:  Old Bernoulli params [B, d] (frozen).

        Returns:
            ratio: [B]
        """
        log_ratio = self.log_prob(a_bin, q_new) - self.log_prob(a_bin, q_old)
        return torch.exp(log_ratio.clamp(-20.0, 20.0))

    def entropy(self, q: Tensor) -> Tensor:
        """
        Shannon entropy of factorized Bernoulli: sum_i H(Bernoulli(q^i)).

        H = sum_i [-q^i log q^i - (1-q^i) log(1-q^i)]

        Used in entropy regularization for bang-bang policy (Eq. (22), Section 4.1.2).

        Args:
            q: Bernoulli parameters [B, d].

        Returns:
            entropy: [B]
        """
        eps = 1e-8
        q_c = q.clamp(eps, 1 - eps)
        h = -(q_c * torch.log(q_c) + (1 - q_c) * torch.log(1 - q_c))
        return h.sum(dim=-1)  # [B]

    def __repr__(self) -> str:
        return f"BangBangPolicy(d={self.d})"
