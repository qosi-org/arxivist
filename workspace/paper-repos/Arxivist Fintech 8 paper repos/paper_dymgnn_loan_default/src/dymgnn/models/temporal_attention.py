"""
models/temporal_attention.py — Temporal Attention over Time Snapshots.

Implements the attention mechanism from Section 3.5, Equations 16–18:
    s^(t)   = a_h H^(t) W_h                          (Eq. 16)
    β^(t)   = softmax over τ of exp(s^(t))            (Eq. 17)
    H_att   = Σ_{t=1}^{τ} β^(t) H^(t)               (Eq. 18)

Assigns different importance to each time snapshot, allowing the model
to focus on the most informative periods. Section 5 shows attention weights
rise sharply near timestamp 6 (most recent), consistent with recency bias.

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
Section 3.5 and Figure 4/9.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalAttention(nn.Module):
    """Soft attention mechanism over τ time snapshots (Section 3.5).

    Computes a weighted average of RNN hidden states across all snapshots,
    with weights determined by a learned scoring function.

    The paper reports (Fig. 9) that attention scores increase progressively
    from timestamp 1 to τ=6, peaking just below 0.6 at t=6, confirming
    that recent information is most important for default prediction.

    Args:
        hidden_dim: RNN hidden state dimension D.
        num_nodes: Number of nodes in one snapshot nl (used for a_h shape).
            Note: a_h ∈ R^{1×nl}, W_h ∈ R^{D×1} per Eq. 16.
    """

    def __init__(self, hidden_dim: int, num_nodes: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # a_h ∈ R^{1×nl} — node-level attention weight vector (Eq. 16)
        self.a_h = nn.Parameter(torch.randn(1, num_nodes))
        # W_h ∈ R^{D×1} — hidden-state projection (Eq. 16)
        self.W_h = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_seq: Tensor) -> tuple[Tensor, Tensor]:
        """Compute attention-weighted aggregation over time snapshots.

        Implements Eq. 16–18.

        Args:
            hidden_seq: Sequence of RNN hidden states [tau, nl, D].

        Returns:
            H_att: Attention-weighted aggregated embedding [nl, D].
            betas: Normalized attention weights [tau, nl] for interpretability.
        """
        assert hidden_seq.dim() == 3, (
            f"Expected [tau, nl, D], got {hidden_seq.shape}"
        )
        tau, nl, D = hidden_seq.shape
        assert D == self.hidden_dim, (
            f"hidden_dim mismatch: expected {self.hidden_dim}, got {D}"
        )

        # Eq. 16: s^(t) = a_h H^(t) W_h
        # a_h: [1, nl], H^(t): [nl, D], W_h: [D, 1]
        # s^(t) = (a_h [1,nl]) @ (H^(t) [nl,D] @ W_h [D,1]) = scalar per t
        # Per Eq. 16, s^(t) is a scalar (a_h ∈ R^{1×nl}, W_h ∈ R^{D×1})
        scores = []
        for t in range(tau):
            Ht = hidden_seq[t]              # [nl, D]
            s_t = self.a_h @ self.W_h(Ht)  # [1, nl] @ [nl, 1] = [1, 1] scalar
            scores.append(s_t.squeeze())    # scalar

        scores_t = torch.stack(scores, dim=0)  # [tau]

        # Eq. 17: β^(t) = softmax over τ
        betas = F.softmax(scores_t, dim=0)     # [tau]

        # Eq. 18: H_att = Σ β^(t) H^(t)
        H_att = torch.zeros(nl, D, device=hidden_seq.device)
        for t in range(tau):
            H_att += betas[t] * hidden_seq[t]  # [nl, D]

        # Expand betas for interpretability (broadcast over nodes)
        betas_expanded = betas.unsqueeze(1).expand(tau, nl)  # [tau, nl]
        return H_att, betas_expanded

    def __repr__(self) -> str:
        return (
            f"TemporalAttention(hidden_dim={self.hidden_dim}, "
            f"num_nodes={self.num_nodes})"
        )
