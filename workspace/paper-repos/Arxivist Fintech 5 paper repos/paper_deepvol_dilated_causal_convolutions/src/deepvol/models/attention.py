"""
Bahdanau (additive) attention mechanism.
Section 4.2: "we use the classic attention mechanism (Bahdanau et al. 2014)"
NOT self-attention / Transformer attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Additive attention that collapses the temporal dimension.
    Weights each time-step's contribution to the final context vector.

    Reference: Bahdanau et al. 2014, "Neural Machine Translation by
    Jointly Learning to Align and Translate"

    Args:
        hidden_dim: Number of channels in the skip-summed tensor (skip_channels)
        attention_dim: Internal projection dimension (ASSUMED = hidden_dim)
    """
    def __init__(self, hidden_dim: int, attention_dim: int | None = None):
        super().__init__()
        attention_dim = attention_dim or hidden_dim
        self.W = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]  — skip-summed tensor over time
        Returns:
            context: [B, C]  — attended context vector
        """
        assert x.dim() == 3, f"Expected [B, C, L], got {x.shape}"
        # Transpose to [B, L, C] for linear layers
        x_t = x.permute(0, 2, 1)                   # [B, L, C]
        energy = torch.tanh(self.W(x_t))            # [B, L, attention_dim]
        scores = self.v(energy).squeeze(-1)          # [B, L]
        weights = F.softmax(scores, dim=-1)          # [B, L]
        context = torch.bmm(weights.unsqueeze(1), x_t).squeeze(1)  # [B, C]
        return context

    def __repr__(self):
        return f"BahdanauAttention(hidden_dim={self.W.in_features})"
