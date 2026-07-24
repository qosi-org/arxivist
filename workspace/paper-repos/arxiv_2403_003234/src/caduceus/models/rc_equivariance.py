"""Pure-PyTorch reference for the Caduceus RC-equivariance components.

These implement the paper's core architectural contributions (Sec 3-4) in
plain PyTorch so that the RC-equivariance property (Theorem 3.1) can be
*unit-tested on CPU* without the fused Mamba CUDA kernels. The actual
fine-tuning path (`classifier.py`) loads the official pretrained Caduceus from
the HuggingFace Hub, whose forward pass uses the real `mamba-ssm` kernels.

Key operations (Sec 3.2, Appendix A):

    RC(X^{1:D}_{1:T}) := X^{D:1}_{T:1}      # reverse along length AND channels

    split(X) := [X_a, X_b]                  # halve along channel dim

    MambaDNA(X) := concat( M(X_a), RC( M( RC(X_b) ) ) )

with the sequence operator M (Mamba or BiMamba) *shared* between the two
halves. The `flip_chan` operator reverses only the channel dim.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def reverse_complement_tensor(x: torch.Tensor) -> torch.Tensor:
    """RC operator on a hidden tensor [B, L, D]: reverse length and channels.

    Implements ``RC(X) = X^{D:1}_{T:1}`` from Eq (6). For a *hidden* tensor the
    "complement" is realised as reversing the channel axis (the learned analogue
    of A<->T, C<->G on one-hot inputs), matching the paper's group action.
    """
    assert x.dim() == 3, f"expected [B, L, D], got {tuple(x.shape)}"
    return torch.flip(x, dims=(1, 2))


def flip_chan(x: torch.Tensor) -> torch.Tensor:
    """flip_chan(X) := X^{D:1} — reverse only the channel dimension."""
    return torch.flip(x, dims=(-1,))


class _SimpleMamba(nn.Module):
    """A lightweight stand-in for the Mamba block (CPU-friendly).

    NOT the real selective-SSM — it is a causal depthwise-conv + gated MLP that
    preserves the [B, L, D] -> [B, L, D] contract and is deterministic, so the
    *equivariance algebra* around it (BiMamba / MambaDNA) can be verified
    exactly. The official model swaps this for `mamba_ssm.Mamba`.
    """

    def __init__(self, dim: int, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.dim = dim
        inner = expand * dim
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, kernel_size=d_conv, groups=inner,
                              padding=d_conv - 1, bias=True)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        xz = self.in_proj(x)                      # [B, L, 2*inner]
        xh, z = xz.chunk(2, dim=-1)
        xh = xh.transpose(1, 2)                    # [B, inner, L]
        xh = self.conv(xh)[..., :l]               # causal conv
        xh = xh.transpose(1, 2)
        xh = self.act(xh) * self.act(z)           # gated
        return self.out_proj(xh)


class BiMamba(nn.Module):
    """Parameter-efficient bi-directional Mamba (Sec 3.1, Fig 1 middle).

    Apply a *single* shared Mamba to the sequence and to its length-reversed
    copy; flip the reverse output back and add:

        y = M(x) + flip_L( M( flip_L(x) ) )

    Weight tying (one `M`) keeps the parameter count at ~1x rather than 2x.
    """

    def __init__(self, dim: int, mamba: nn.Module | None = None) -> None:
        super().__init__()
        self.mamba = mamba if mamba is not None else _SimpleMamba(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.mamba(x)
        rev = torch.flip(self.mamba(torch.flip(x, dims=(1,))), dims=(1,))
        return fwd + rev


class MambaDNA(nn.Module):
    """RC-equivariant Mamba module (Sec 3.2, Eq 8; Appendix A).

        MambaDNA(X) = concat( M(X_a), RC( M( RC(X_b) ) ) )

    where [X_a, X_b] = split(X) along channels and M (here BiMamba) is shared.
    Satisfies ``RC o MambaDNA = MambaDNA o RC`` (Theorem 3.1), verified in tests.
    """

    def __init__(self, dim: int, inner: nn.Module | None = None) -> None:
        super().__init__()
        assert dim % 2 == 0, "channel dim must be even (channel split)"
        self.dim = dim
        half = dim // 2
        # One shared sequence operator applied to each D/2 half.
        self.inner = inner if inner is not None else BiMamba(half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        x_a, x_b = x[..., :half], x[..., half:]
        y_a = self.inner(x_a)
        y_b = reverse_complement_tensor(self.inner(reverse_complement_tensor(x_b)))
        return torch.cat([y_a, y_b], dim=-1)


class RCEquivariantEmbedding(nn.Module):
    """RC-equivariant token embedding (Sec 4.1).

        Emb_RCe(X) = concat( Emb(X), RC( Emb( RC(X) ) ) )

    where Emb projects one-hot nucleotides to D/2 channels. RC on the discrete
    input reverses length and complements the 4 bases (A<->T, C<->G).
    """

    _COMPLEMENT = {0: 3, 1: 2, 2: 1, 3: 0}  # A<->T (0,3), C<->G (1,2); others identity

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.emb = nn.Embedding(vocab_size, dim // 2)
        comp = torch.arange(vocab_size)
        for k, v in self._COMPLEMENT.items():
            if k < vocab_size and v < vocab_size:
                comp[k] = v
        self.register_buffer("complement", comp)

    def _rc_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.complement[torch.flip(ids, dims=(1,))]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(ids)                                  # [B, L, D/2]
        e_rc = torch.flip(self.emb(self._rc_ids(ids)), dims=(1, 2))
        return torch.cat([e, e_rc], dim=-1)                # [B, L, D]


class RCEquivariantLMHead(nn.Module):
    """RC-equivariant language-model head (Sec 4.1).

        LM_RCe(X) = LM(X_a) + flip_chan( LM(X_b) )

    with [X_a, X_b] the channel split and a single shared linear map LM.
    """

    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.lm = nn.Linear(dim // 2, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        half = h.size(-1) // 2
        h_a, h_b = h[..., :half], h[..., half:]
        return self.lm(h_a) + flip_chan(self.lm(h_b))
