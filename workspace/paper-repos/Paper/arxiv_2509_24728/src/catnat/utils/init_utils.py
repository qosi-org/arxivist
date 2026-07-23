"""
Weight and parameter initialization utilities.

Implements initialization schemes specified in the paper:
  - GSL score init: U(0, 0.1) — Appendix E.5
  - RL CNN orthogonal init — Appendix I.2
"""

import torch
import torch.nn as nn
from torch import Tensor


def init_scores_uniform(shape: tuple, low: float = 0.0, high: float = 0.1) -> Tensor:
    """Initialize GSL Bernoulli edge scores from a uniform distribution.

    Paper: Appendix E.5 — "Scores were initialized so that θ_ij ~ U(0, 0.1)"
    Confidence: 0.99 (explicitly stated).

    Args:
        shape: Shape of the score tensor (e.g. [n_edges] or [n_nodes, n_nodes]).
        low: Lower bound of uniform distribution. Default 0.0.
        high: Upper bound of uniform distribution. Default 0.1.

    Returns:
        Tensor of given shape sampled from U(low, high).
    """
    return torch.empty(shape).uniform_(low, high)


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Apply orthogonal initialization to all Linear and Conv2d layers.

    Paper: Appendix I.2 — "All layers are initialized using orthogonal initialization."
    Confidence: 0.97 (explicitly stated).

    Used for the AtariCNN backbone in the RL experiment.

    Args:
        module: An nn.Module (applied recursively to all submodules).
        gain: Scaling factor for orthogonal init. Default 1.0;
              use sqrt(2) for ReLU activations (common practice).

    Returns:
        The same module, with weights initialized in-place.
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module
