"""
CatNat: A Natural Parameterization for Categorical Random Variables.

Implements the catnat function π: R^{K-1} → Δ^{K-1} that maps K-1
unnormalized scores to a valid K-class categorical probability vector
via a hierarchical binary tree of depth H = log2(K).

Key theoretical properties (proved in the paper):
  - The Fisher Information Matrix is DIAGONAL (Theorem 4.2)
  - Using NaturalActivation ν, diagonal entries depend only on P(a_i) (Corollary 4.3)
  - Cross-score coupling in the pulled-back FIM w.r.t. θ vanishes (Section 4.2.3, Eq. 14-15)

Paper: Section 4.2, Eqs. 7–13, Theorems 4.2, Corollary 4.3.
arXiv: 2509.24728v2. ICML 2026.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .activations import build_activation
from .utils.tree_utils import BinaryTreeIndex


class CatNat(nn.Module):
    """Catnat parameterization of a categorical distribution.

    Replaces the softmax function with a sequence of hierarchical binary decisions.
    Each of the K-1 internal tree nodes receives one unnormalized score s_i and
    produces a Bernoulli probability a_i = activation(s_i). Leaf probabilities
    are products of Bernoulli factors along the path from root to leaf (Eq. 8).

    This is a STATELESS transform (no learnable parameters inside CatNat itself).
    The learnable scores s are held by the calling model (e.g. VAEEncoder, GSLModel).

    Args:
        K:          Number of categories. Must be a power of 2.
        activation: Activation function name: "natural" (recommended) or "sigmoid".
        C:          Centre for NaturalActivation. Default 0.0.
        A:          Slope for NaturalActivation. Default 2π.

    Input:  s  of shape [..., K-1]  (unnormalized scores)
    Output: p  of shape [..., K]    (categorical probabilities summing to 1)

    Example:
        >>> catnat = CatNat(K=8, activation="natural")
        >>> s = torch.randn(4, 7)       # batch of 4, K-1=7 scores
        >>> p = catnat(s)               # [4, 8], sums to 1 along dim=-1
    """

    def __init__(
        self,
        K: int,
        activation: str = "natural",
        C: float = 0.0,
        A: float = 2 * math.pi,
    ) -> None:
        super().__init__()
        self.K = K
        self.H = int(math.log2(K))   # tree depth

        # Build activation function (NaturalActivation or SigmoidActivation)
        self.activation = build_activation(activation, C=C, A=A)

        # Precompute tree index buffers (non-learnable)
        self._tree = BinaryTreeIndex(K)
        # Register as buffers so they move with .to(device) and appear in state_dict
        self.register_buffer("_ancestor_mask",  self._tree.ancestor_mask.float())   # [K, K-1]
        self.register_buffer("_branch_taken",   self._tree.branch_taken.float())    # [K, K-1]
        self.register_buffer("_leaf_paths",     self._tree.leaf_paths)              # [K, H]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, s: Tensor) -> Tensor:
        """Map unnormalized scores to categorical probabilities.

        Implements Eq. 8: p_{b1,...,bH} = ∏_h a_{[b1,...,b_{h-1}]}^{b_h} * (1-a_{...})^{1-b_h}
        Computation is done in log-space for numerical stability (RISK-04).

        Args:
            s: Unnormalized scores, shape [..., K-1].

        Returns:
            Probability vector, shape [..., K], summing to 1 along the last dim.
        """
        assert s.shape[-1] == self.K - 1, (
            f"CatNat(K={self.K}) expects input shape [..., {self.K-1}], got {s.shape}"
        )
        return self.log_prob(s).exp()

    def log_prob(self, s: Tensor) -> Tensor:
        """Return log-probabilities of all K categories.

        Log-space implementation of Eq. 8. Preferred for numerical stability.

        Vectorized forward pass (no Python loops):
          1. Compute node activations a_i = activation(s_i)            [..., K-1]
          2. Compute log(a_i) and log(1 - a_i)                         [..., K-1]
          3. For each leaf k: sum log(a_i) or log(1-a_i) for ancestors  [..., K]
             using precomputed ancestor_mask and branch_taken buffers.

        Paper: Eq. 8, Section 4.2.1.

        Args:
            s: Unnormalized scores, shape [..., K-1].

        Returns:
            Log-probabilities, shape [..., K].
        """
        # Step 1: node Bernoulli probabilities — Eq. 8, a_i = activation(s_i)
        a = self.activation(s)                                   # [..., K-1]

        # Step 2: log of both branches — clamp for numerical safety (RISK-04)
        eps = 1e-8
        log_a       = torch.log(a.clamp(min=eps))                # [..., K-1]
        log_1_minus_a = torch.log((1.0 - a).clamp(min=eps))     # [..., K-1]

        # Step 3: for each leaf, sum log-factors over its ancestors
        # branch_taken[k, i] ∈ {0,1}: direction at node i toward leaf k
        # ancestor_mask[k, i] ∈ {0,1}: 1 iff node i is an ancestor of leaf k
        #
        # Eq. 8 in log-space:
        #   log p_k = Σ_i  ancestor_mask[k,i] *
        #               (branch_taken[k,i]*log(a_i) + (1-branch_taken[k,i])*log(1-a_i))
        #
        # Shape broadcasting:
        #   s:               [..., K-1]
        #   log_a:           [..., K-1]
        #   _branch_taken:   [K, K-1]   (broadcast over ... batch dims)
        #   _ancestor_mask:  [K, K-1]

        # Select the correct log-factor per (leaf, node) pair
        # branch_taken=1 → use log(a), branch_taken=0 → use log(1-a)
        # [..., K-1] → unsqueeze to [..., 1, K-1] for broadcasting over K leaves
        log_a_exp         = log_a.unsqueeze(-2)           # [..., 1, K-1]
        log_1_minus_a_exp = log_1_minus_a.unsqueeze(-2)   # [..., 1, K-1]

        # [K, K-1] branch_taken selects which log to use at each (leaf, node)
        branch = self._branch_taken  # [K, K-1]
        log_factor = branch * log_a_exp + (1.0 - branch) * log_1_minus_a_exp  # [..., K, K-1]

        # Zero out contributions from non-ancestor nodes
        log_factor = log_factor * self._ancestor_mask      # [..., K, K-1]

        # Sum over nodes dimension → log p_k for each leaf k
        log_p = log_factor.sum(dim=-1)                     # [..., K]
        return log_p

    def node_reach_probs(self, s: Tensor) -> Tensor:
        """Compute P(a_i) — the probability of reaching each internal node from the root.

        P(a_i) = sum of p_k for all leaves k that are descendants of node i.
        Equivalently, it is the product of Bernoulli factors along the path
        from root to node i (Eq. 9).

        Used in FIM diagonal computation (Theorem 4.2, Eq. 11).

        Args:
            s: Scores, shape [..., K-1].

        Returns:
            Reach probabilities, shape [..., K-1].
        """
        # p: [..., K]
        p = self.forward(s)
        # P(a_i) = sum of p_k for descendants of node i
        # ancestor_mask[k, i] = 1 iff node i is ancestor of leaf k
        # So  P(a_i) = sum_k p_k * ancestor_mask[k, i]
        # p: [..., K], ancestor_mask: [K, K-1]
        reach = torch.einsum("...k,ki->...i", p, self._ancestor_mask)  # [..., K-1]
        return reach

    def fim_diagonal(self, s: Tensor) -> Tensor:
        """Compute the diagonal entries of the FIM per Theorem 4.2 (Eq. 11).

        G_a(s)_ii = P(a_i) * (∂a_i/∂s_i)² * 1/(a_i*(1-a_i))

        For diagnostic and analysis purposes only — not used during training.

        Args:
            s: Scores, shape [..., K-1].

        Returns:
            Diagonal FIM entries, shape [..., K-1].
        """
        p_reach = self.node_reach_probs(s)        # [..., K-1]
        # Delegate to activation's FIM diagnostic method
        return self.activation.fim_diagonal_factor(s, p_reach)

    def __repr__(self) -> str:
        return (
            f"CatNat(K={self.K}, H={self.H}, activation={self.activation})"
        )


# ---------------------------------------------------------------------------
# Baseline parameterizations
# ---------------------------------------------------------------------------

class SoftmaxParam(nn.Module):
    """Standard softmax parameterization — baseline.

    p_i = exp(s_i) / sum_k exp(s_k)

    Paper: Section 4.1, Eq. 5. Has a DENSE Fisher Information Matrix (Proposition 4.1, Eq. 6).

    Args:
        dim: Dimension to apply softmax over. Default -1.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, s: Tensor) -> Tensor:
        """Apply softmax.

        Args:
            s: Scores, shape [..., K].

        Returns:
            Probabilities, shape [..., K].
        """
        return torch.softmax(s, dim=self.dim)

    def __repr__(self) -> str:
        return f"SoftmaxParam(dim={self.dim})"


class SparseMaxParam(nn.Module):
    """Sparsemax parameterization — baseline.

    Projects scores onto the probability simplex via Euclidean projection.
    Reference: Martins & Astudillo (2016). Paper: Section 5 (evaluated as baseline).

    Implements the O(K log K) sort-based algorithm from Martins & Astudillo (2016).
    """

    def forward(self, s: Tensor) -> Tensor:
        """Apply sparsemax.

        Args:
            s: Scores, shape [..., K].

        Returns:
            Sparse probability vector on the simplex, shape [..., K].
        """
        # Sort descending along last dim
        z = s
        K = z.shape[-1]
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)

        # Cumulative sum for threshold computation
        cumsum = torch.cumsum(z_sorted, dim=-1)
        k_range = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
        # Shape: [..., K]
        threshold_values = (cumsum - 1.0) / k_range
        # Find the valid support: z_sorted[k] > threshold_values[k]
        support = z_sorted > threshold_values  # [..., K]
        # τ(z) = (sum_{i in support} z_i - 1) / |support|
        k_z = support.sum(dim=-1, keepdim=True).float()    # [..., 1]
        tau = (support.float() * z_sorted).sum(dim=-1, keepdim=True) - 1.0
        tau = tau / k_z
        return torch.clamp(z - tau, min=0.0)

    def __repr__(self) -> str:
        return "SparseMaxParam()"


def build_parameterization(name: str, K: int, **catnat_kwargs) -> nn.Module:
    """Factory: return the correct parameterization module by name.

    Args:
        name: One of "natural", "sigmoid", "softmax", "sparsemax".
        K:    Number of categories.
        **catnat_kwargs: Passed to CatNat for catnat variants.

    Returns:
        An nn.Module implementing the parameterization.
    """
    if name == "natural":
        return CatNat(K=K, activation="natural", **catnat_kwargs)
    elif name == "sigmoid":
        return CatNat(K=K, activation="sigmoid", **catnat_kwargs)
    elif name == "softmax":
        return SoftmaxParam(dim=-1)
    elif name == "sparsemax":
        return SparseMaxParam()
    else:
        raise ValueError(
            f"Unknown parameterization '{name}'. "
            "Must be one of: 'natural', 'sigmoid', 'softmax', 'sparsemax'."
        )
