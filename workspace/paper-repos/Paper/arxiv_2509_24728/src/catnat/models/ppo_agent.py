"""
PPO Agent with catnat policy head for discrete Atari environments.

Implements the shared actor-critic architecture from Section 5.3, Appendix I.2.

Architecture (explicitly stated, confidence: 0.97):
  Shared backbone: Conv2d(4,32,8,4) → Conv2d(32,64,4,2) → Conv2d(64,64,3,1) → FC-512
  Policy head:  Linear(512, K-1) → CatNat → Categorical distribution
  Value head:   Linear(512, 1)

All layers use orthogonal initialization (Appendix I.2).
The policy head outputs K-1 scores for catnat, or K scores for softmax.

Paper: Section 5.3, Appendix I.2. arXiv: 2509.24728v2. ICML 2026.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from ..catnat import build_parameterization
from ..utils.config import RLConfig
from ..utils.init_utils import orthogonal_init


class AtariCNN(nn.Module):
    """Shared convolutional backbone for Atari observations.

    Processes stacked grayscale frames into a 512-dimensional feature vector.

    Architecture (Appendix I.2, explicitly stated, confidence: 0.97):
      Conv2d(4,  32, kernel=8, stride=4) → ReLU
      Conv2d(32, 64, kernel=4, stride=2) → ReLU
      Conv2d(64, 64, kernel=3, stride=1) → ReLU
      Flatten → Linear(64*7*7, 512) → ReLU

    Input: [B, 4, 84, 84] (4 stacked 84×84 grayscale frames, pixel values /255)

    Args:
        config: RLModelConfig with conv layer specs.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Layer 1 — explicitly stated: 32 filters, 8×8, stride 4
            nn.Conv2d(4, config.conv1_filters, kernel_size=config.conv1_kernel,
                      stride=config.conv1_stride),
            nn.ReLU(),
            # Layer 2 — explicitly stated: 64 filters, 4×4, stride 2
            nn.Conv2d(config.conv1_filters, config.conv2_filters,
                      kernel_size=config.conv2_kernel, stride=config.conv2_stride),
            nn.ReLU(),
            # Layer 3 — explicitly stated: 64 filters, 3×3, stride 1
            nn.Conv2d(config.conv2_filters, config.conv3_filters,
                      kernel_size=config.conv3_kernel, stride=config.conv3_stride),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Conv output size for 84×84 input: 64 * 7 * 7 = 3136
        conv_out = config.conv3_filters * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(conv_out, config.fc_hidden),
            nn.ReLU(),
        )
        self.output_dim = config.fc_hidden

    def forward(self, obs: Tensor) -> Tensor:
        """Encode observation to feature vector.

        Args:
            obs: Stacked frames, shape [B, 4, 84, 84], dtype uint8 or float32.

        Returns:
            Feature vector, shape [B, 512].
        """
        assert obs.shape[1:] == (4, 84, 84), (
            f"AtariCNN expects [B, 4, 84, 84], got {obs.shape}"
        )
        # Normalize pixel values to [0, 1]
        x = obs.float() / 255.0
        return self.fc(self.conv(x))

    def __repr__(self) -> str:
        return f"AtariCNN(output_dim={self.output_dim})"


class PPOActor(nn.Module):
    """Policy head: features → action distribution via catnat or softmax.

    For catnat: Linear(512, K-1) → CatNat(K) → Categorical(probs)
    For softmax: Linear(512, K)  → Softmax  → Categorical(probs)

    Paper: Appendix I.2 — "policy head that outputs scores for the action distribution."

    Args:
        feature_dim:  Input feature dimension (512).
        n_actions:    Number of discrete actions K.
        param_name:   Parameterization name: "natural" or "softmax".
        catnat_kwargs: Passed to CatNat if using catnat.
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        param_name: str = "natural",
        **catnat_kwargs,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.param_name = param_name

        is_catnat = param_name in ("natural", "sigmoid")
        # K-1 output scores for catnat, K for softmax/sparsemax
        self.scores_out = n_actions - 1 if is_catnat else n_actions

        self.linear = nn.Linear(feature_dim, self.scores_out)
        self.pi = build_parameterization(param_name, K=n_actions, **catnat_kwargs)

    def forward(self, features: Tensor) -> Categorical:
        """Compute action distribution from features.

        Args:
            features: Shape [B, feature_dim].

        Returns:
            torch.distributions.Categorical over n_actions.
        """
        s = self.linear(features)       # [B, K-1] or [B, K]
        if hasattr(self.pi, "forward"):
            probs = self.pi(s)          # [B, K]
        return Categorical(probs=probs)

    def __repr__(self) -> str:
        return (
            f"PPOActor(n_actions={self.n_actions}, "
            f"parameterization='{self.param_name}')"
        )


class PPOCritic(nn.Module):
    """Value head: features → scalar state-value estimate.

    Paper: Appendix I.2 — "value head that estimates the state-value function."

    Args:
        feature_dim: Input feature dimension (512).
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Estimate state value.

        Args:
            features: Shape [B, feature_dim].

        Returns:
            State values, shape [B, 1].
        """
        return self.linear(features)

    def __repr__(self) -> str:
        return "PPOCritic()"


class PPOAgent(nn.Module):
    """Full PPO agent with shared CNN backbone + actor + critic heads.

    Shared-parameter architecture: the CNN backbone is shared between
    the actor (policy) and critic (value) heads, following Huang et al. (2022)
    which is the PPO implementation the paper uses (Appendix I).

    Paper: Section 5.3, Appendix I.2. Confidence: 0.95.

    Args:
        config: RLConfig dataclass.
        n_actions: Number of discrete actions in the environment.
    """

    def __init__(self, config: RLConfig, n_actions: int) -> None:
        super().__init__()
        self.config = config
        self.n_actions = n_actions

        # Shared CNN backbone
        self.backbone = AtariCNN(config.model)

        # Actor (policy) head
        catnat_kwargs = {
            "C": config.catnat.natural_activation_C,
            "A": config.catnat.natural_activation_A,
        } if config.parameterization in ("natural", "sigmoid") else {}
        self.actor = PPOActor(
            feature_dim=self.backbone.output_dim,
            n_actions=n_actions,
            param_name=config.parameterization,
            **catnat_kwargs,
        )

        # Critic (value) head
        self.critic = PPOCritic(feature_dim=self.backbone.output_dim)

        # Orthogonal initialization — Appendix I.2, explicitly stated
        orthogonal_init(self, gain=(2 ** 0.5))  # sqrt(2) for ReLU layers

    def get_action_and_value(self, obs: Tensor) -> Dict[str, Tensor]:
        """Sample action and compute log-prob, entropy, and value.

        Used during rollout collection.

        Args:
            obs: Observations, shape [B, 4, 84, 84].

        Returns:
            Dict with keys:
              'action':   Sampled actions [B]
              'log_prob': Log-probability of sampled action [B]
              'entropy':  Policy entropy [B]
              'value':    State-value estimate [B, 1]
        """
        features = self.backbone(obs)
        dist = self.actor(features)
        action = dist.sample()
        return {
            "action":   action,
            "log_prob": dist.log_prob(action),
            "entropy":  dist.entropy(),
            "value":    self.critic(features),
        }

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Dict[str, Tensor]:
        """Evaluate log-probabilities and values for given obs/action pairs.

        Used during PPO update step (to compute probability ratio r_t(θ)).

        Args:
            obs:     Observations, shape [B, 4, 84, 84].
            actions: Actions taken, shape [B].

        Returns:
            Dict with keys:
              'log_prob': Log-prob of given actions under current policy [B]
              'entropy':  Policy entropy [B]
              'value':    State-value estimate [B, 1]
        """
        features = self.backbone(obs)
        dist = self.actor(features)
        return {
            "log_prob": dist.log_prob(actions),
            "entropy":  dist.entropy(),
            "value":    self.critic(features),
        }

    def __repr__(self) -> str:
        return (
            f"PPOAgent(n_actions={self.n_actions}, "
            f"parameterization='{self.config.parameterization}')"
        )
