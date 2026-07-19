"""Smoke tests: verify each model family builds and runs a forward pass on random data."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sert_asset_pricing.models.transformer_variants import build_model  # noqa: E402
from sert_asset_pricing.models.positional_encoding import SinusoidalPositionalEncoding  # noqa: E402
from sert_asset_pricing.models.mlp_autoencoder import MLPAutoencoder  # noqa: E402
from sert_asset_pricing.models.attention import CausalMask, MultiHeadSelfAttention  # noqa: E402

BASE_CFG = {
    "model": {
        "input_factor_dim": 16,
        "d_model": 32,
        "num_heads": 4,
        "num_blocks": 1,
        "ffn_hidden_ratio": 0.7,
        "ffn_hidden_ratio_lnf": 0.2,
        "pretrain_hidden_layers": 1,
        "activation": "relu",
        "dropout": 0.0,
        "layer_norm_first": False,
    }
}


@pytest.mark.parametrize(
    "family,needs_teacher_forcing",
    [
        ("pretrained_transformer", True),
        ("pretrained_transformer_lnf", True),
        ("standard_transformer", True),
        ("sert", False),
        ("sert_lnf", False),
        ("encoder_only_transformer", False),
    ],
)
def test_model_builds_and_forwards(family, needs_teacher_forcing):
    cfg = {"model": dict(BASE_CFG["model"])}
    cfg["model"]["family"] = family
    cfg["model"]["layer_norm_first"] = "lnf" in family
    model = build_model(cfg)
    x = torch.randn(2, 10, cfg["model"]["input_factor_dim"])
    if needs_teacher_forcing:
        y_shifted = torch.randn(2, 10, 1)
        out = model(x, y_shifted)
    else:
        out = model(x)
    assert out.shape == (2, 10, 1)


def test_positional_encoding_shape():
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=50)
    x = torch.zeros(3, 10, 16)
    out = pe(x)
    assert out.shape == (3, 10, 16)


def test_mlp_autoencoder_projects_dims():
    ae = MLPAutoencoder(in_dim=8, out_dim=20, hidden_ratio=0.7)
    x = torch.randn(4, 5, 8)
    out = ae(x)
    assert out.shape == (4, 5, 20)


def test_causal_mask_upper_triangular_is_masked():
    mask = CausalMask.build(seq_len=5, device=torch.device("cpu"))
    assert torch.isneginf(mask[0, 1])
    assert mask[1, 0] == 0
    assert mask[4, 4] == 0


def test_self_attention_respects_causal_mask_shape():
    attn = MultiHeadSelfAttention(d_model=16, num_heads=4, dropout=0.0)
    x = torch.randn(2, 7, 16)
    mask = CausalMask.build(7, torch.device("cpu"))
    out = attn(x, mask)
    assert out.shape == (2, 7, 16)
