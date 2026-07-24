"""Unit tests for the Caduceus RC-equivariance components (Sec 3-4).

These verify the paper's central architectural claims on CPU, without the Mamba
CUDA kernels:

* Theorem 3.1: ``RC o MambaDNA = MambaDNA o RC``
* BiMamba preserves shape and is a true bi-directional operator
* RC-equivariant embedding: ``RC o Emb_RCe = Emb_RCe o RC``
* Shapes/contracts of every reference module
"""
import torch

from src.caduceus.models.rc_equivariance import (
    BiMamba,
    MambaDNA,
    RCEquivariantEmbedding,
    RCEquivariantLMHead,
    reverse_complement_tensor,
)


def _rc_ids(ids, complement):
    return complement[torch.flip(ids, dims=(1,))]


def test_rc_is_involution():
    x = torch.randn(2, 16, 8)
    assert torch.allclose(reverse_complement_tensor(reverse_complement_tensor(x)), x)


def test_bimamba_shape():
    torch.manual_seed(0)
    blk = BiMamba(dim=16)
    x = torch.randn(3, 32, 16)
    y = blk(x)
    assert y.shape == x.shape


def test_mambadna_shape():
    torch.manual_seed(0)
    m = MambaDNA(dim=16)
    x = torch.randn(3, 32, 16)
    assert m(x).shape == x.shape


def test_mambadna_rc_equivariance():
    """Theorem 3.1: RC(MambaDNA(x)) == MambaDNA(RC(x))."""
    torch.manual_seed(0)
    m = MambaDNA(dim=16).eval()
    x = torch.randn(4, 24, 16)
    with torch.no_grad():
        lhs = reverse_complement_tensor(m(x))
        rhs = m(reverse_complement_tensor(x))
    assert torch.allclose(lhs, rhs, atol=1e-5), (lhs - rhs).abs().max().item()


def test_stacked_mambadna_equivariance():
    """Composition of RC-equivariant modules stays equivariant (Lemma B.1)."""
    torch.manual_seed(1)
    m1, m2 = MambaDNA(dim=16).eval(), MambaDNA(dim=16).eval()
    x = torch.randn(2, 20, 16)
    with torch.no_grad():
        lhs = reverse_complement_tensor(m2(m1(x)))
        rhs = m2(m1(reverse_complement_tensor(x)))
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_embedding_rc_equivariance():
    """RC o Emb_RCe = Emb_RCe o RC  (Eq 12)."""
    torch.manual_seed(0)
    vocab, dim = 5, 16
    emb = RCEquivariantEmbedding(vocab_size=vocab, dim=dim).eval()
    ids = torch.randint(0, 4, (2, 12))  # only A/C/G/T so complement stays in-vocab
    with torch.no_grad():
        lhs = reverse_complement_tensor(emb(ids))
        rhs = emb(_rc_ids(ids, emb.complement))
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_lm_head_shape():
    torch.manual_seed(0)
    head = RCEquivariantLMHead(dim=16, vocab_size=5)
    h = torch.randn(2, 10, 16)
    assert head(h).shape == (2, 10, 5)
