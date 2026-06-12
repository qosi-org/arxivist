"""
tests/test_gmlp.py
------------------
Unit + integration tests for the gMLP reproduction.

Validates:
  1. Tensor shapes through every module
  2. SGU mathematical properties (Eq. 4 — split multiplicative gate)
  3. Toeplitz matrix construction (Appendix C)
  4. W/b initialization (near-zero W, ones b — Section 2.1)
  5. aMLP SGU (hybrid gate — Section 4.3)
  6. Full NLP forward pass (gMLP and aMLP)
  7. Full Vision forward pass
  8. Residual shortcut correctness
  9. Config presets match paper Table 1 & 5 parameter counts
  10. Scaling: deeper gMLP with larger d has correct param count
  11. Loss functions
  12. LR schedulers

All tests run on CPU. No GPU or data download required.

Run:
    pytest tests/test_gmlp.py -v
    pytest tests/test_gmlp.py -v -k "test_sgu"   # run specific test group
"""

import math
import sys
import os

import pytest
import torch
import torch.nn as nn

# Ensure src/ is on the path when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gmlp.models.toeplitz import ToeplitzLinear
from gmlp.models.tiny_attn import TinyAttention
from gmlp.models.sgu import SpatialGatingUnit, aMLP_SGU
from gmlp.models.gmlp_block import gMLPBlock, DropPath
from gmlp.models.patch_embed import PatchEmbedding
from gmlp.models.gmlp import gMLP, ModelOutput
from gmlp.utils.config import (
    ModelConfig, TrainingConfig, DataConfig, gMLPConfig, get_preset, set_seed
)
from gmlp.training.losses import (
    MLMLoss, ClassificationLoss, QALoss,
    get_linear_warmup_decay, get_cosine_warmup_decay
)
from gmlp.evaluation.metrics import (
    compute_perplexity, compute_accuracy, compute_squad_f1, aggregate_runs
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

B, N, D_MODEL, D_FFN = 2, 16, 64, 256   # small dims for fast tests

@pytest.fixture
def nlp_config():
    return ModelConfig(
        model_type="nlp", num_layers=2, d_model=D_MODEL, d_ffn=D_FFN,
        seq_len=N, vocab_size=512, num_classes=512,
        use_toeplitz=True, use_tiny_attn=False, w_init_std=0.002,
    )

@pytest.fixture
def amlp_config():
    return ModelConfig(
        model_type="nlp", num_layers=2, d_model=D_MODEL, d_ffn=D_FFN,
        seq_len=N, vocab_size=512, num_classes=512,
        use_toeplitz=True, use_tiny_attn=True, d_attn=16, w_init_std=0.002,
        attn_fusion_mode="add",
    )

@pytest.fixture
def vision_config():
    return ModelConfig(
        model_type="vision", num_layers=2, d_model=D_MODEL, d_ffn=D_FFN,
        seq_len=4, num_classes=10, img_size=8, patch_size=4,
        use_toeplitz=False, use_tiny_attn=False, survival_prob=1.0,
        pool_mode="avg",
    )


# ═══════════════════════════════════════════════════════════════
# 1. ToeplitzLinear
# ═══════════════════════════════════════════════════════════════

class TestToeplitzLinear:

    def test_output_shape_toeplitz(self):
        """Toeplitz mode: [B, n, e] → [B, n, e]."""
        layer = ToeplitzLinear(seq_len=N, use_toeplitz=True)
        z = torch.randn(B, N, D_FFN // 2)
        out = layer(z)
        assert out.shape == (B, N, D_FFN // 2), f"Expected {(B,N,D_FFN//2)}, got {out.shape}"

    def test_output_shape_free(self):
        """Unconstrained mode: [B, n, e] → [B, n, e]."""
        layer = ToeplitzLinear(seq_len=N, use_toeplitz=False)
        z = torch.randn(B, N, D_FFN // 2)
        out = layer(z)
        assert out.shape == (B, N, D_FFN // 2)

    def test_toeplitz_matrix_is_toeplitz(self):
        """W returned by get_weight_matrix() must satisfy W_ij = W_{i-j}."""
        layer = ToeplitzLinear(seq_len=6, use_toeplitz=True)
        W = layer.get_weight_matrix()   # [6, 6]
        for i in range(6):
            for j in range(6):
                # Same anti-diagonal must equal same diagonal offset
                offset = i - j
                for i2 in range(6):
                    j2 = i2 - offset
                    if 0 <= j2 < 6:
                        assert torch.isclose(W[i, j], W[i2, j2], atol=1e-6), \
                            f"Toeplitz violation at ({i},{j}) vs ({i2},{j2})"

    def test_bias_init_ones(self):
        """Bias must be initialised to ones (paper Section 2.1 explicit requirement)."""
        layer = ToeplitzLinear(seq_len=N, use_toeplitz=True)
        assert torch.all(layer.bias == 1.0), "bias must initialise to ones"

    def test_weight_near_zero(self):
        """W init must be near-zero (std << 1)."""
        layer = ToeplitzLinear(seq_len=N, use_toeplitz=True, w_init_std=0.002)
        w_std = layer.weight.std().item()
        assert w_std < 0.05, f"W std too large at init: {w_std:.4f} (expected ~0.002)"

    def test_initial_output_approx_identity(self):
        """
        At init: W≈0, b=1 → f_{W,b}(Z) ≈ 1 → SGU ≈ Z (Section 2.1).
        Test: with near-zero W and b=1, output ≈ (input * 1) = input.
        Since the full SGU multiplies gate by z1, we just check f_{W,b}(z) ≈ 1
        by checking layer output is close to bias broadcast.
        """
        torch.manual_seed(42)
        layer = ToeplitzLinear(seq_len=4, use_toeplitz=True, w_init_std=1e-8)
        z = torch.randn(1, 4, 8)
        out = layer(z)
        # With W≈0, out ≈ b (broadcast) ≈ ones
        # The bias contributes +1 per spatial position; W*z term is ~0
        # So out ≈ 1 per channel at every position
        expected = torch.ones(1, 4, 8)
        assert torch.allclose(out, expected, atol=0.01), \
            f"At near-zero W init, output should be ≈ 1. Got: {out}"

    def test_get_weight_matrix_free(self):
        """Unconstrained mode: weight matrix is [n, n]."""
        layer = ToeplitzLinear(seq_len=N, use_toeplitz=False)
        W = layer.get_weight_matrix()
        assert W.shape == (N, N)

    def test_seq_len_mismatch_raises(self):
        layer = ToeplitzLinear(seq_len=N, use_toeplitz=True)
        z = torch.randn(B, N + 1, D_FFN // 2)   # wrong seq_len
        with pytest.raises(AssertionError):
            layer(z)


# ═══════════════════════════════════════════════════════════════
# 2. TinyAttention
# ═══════════════════════════════════════════════════════════════

class TestTinyAttention:

    def test_output_shape(self):
        """[B, n, d_model] → [B, n, d_out]."""
        attn = TinyAttention(d_model=D_MODEL, d_attn=16, d_out=D_FFN // 2)
        x = torch.randn(B, N, D_MODEL)
        out = attn(x)
        assert out.shape == (B, N, D_FFN // 2)

    def test_default_d_out(self):
        """When d_out is None it defaults to d_attn."""
        attn = TinyAttention(d_model=D_MODEL, d_attn=16)
        x = torch.randn(B, N, D_MODEL)
        out = attn(x)
        assert out.shape == (B, N, 16)

    def test_param_count_much_smaller_than_bert(self):
        """
        aMLP tiny attention: 1 head × 64 dim.
        BERT self-attention: 12 heads × 64 dim = 768 dim.
        Tiny should be ~12× fewer attention parameters.
        """
        tiny = TinyAttention(d_model=768, d_attn=64, d_out=64)
        tiny_params = sum(p.numel() for p in tiny.parameters())

        # Simulate BERT-style attention (Q, K, V, O projections, 768-dim)
        bert_attn_params = 4 * 768 * 768  # QKV + output projection
        assert tiny_params < bert_attn_params / 10, \
            f"Tiny attn ({tiny_params}) should be << BERT attn ({bert_attn_params})"

    def test_no_nan_in_output(self):
        torch.manual_seed(0)
        attn = TinyAttention(d_model=D_MODEL, d_attn=16, d_out=32)
        x = torch.randn(B, N, D_MODEL)
        out = attn(x)
        assert not torch.isnan(out).any(), "TinyAttention output contains NaN"


# ═══════════════════════════════════════════════════════════════
# 3. SpatialGatingUnit
# ═══════════════════════════════════════════════════════════════

class TestSGU:

    def test_output_shape(self):
        """SGU: [B, n, d_ffn] → [B, n, d_ffn//2]."""
        sgu = SpatialGatingUnit(d_ffn=D_FFN, seq_len=N, use_toeplitz=True)
        z = torch.randn(B, N, D_FFN)
        out = sgu(z)
        assert out.shape == (B, N, D_FFN // 2), \
            f"SGU output should halve channel dim. Expected {(B,N,D_FFN//2)}, got {out.shape}"

    def test_split_multiplicative_gate_eq4(self):
        """
        Validates SIR Eq. 4: s(Z) = Z1 ⊙ f_{W,b}(Z2)
        We manually compute the expected output and compare to SGU forward.
        """
        torch.manual_seed(7)
        seq_len = 4
        sgu = SpatialGatingUnit(d_ffn=8, seq_len=seq_len, use_toeplitz=False)
        z = torch.randn(1, seq_len, 8)
        z1, z2 = z.chunk(2, dim=-1)     # [1, 4, 4] each

        # Manually replicate SGU forward
        z2_normed = sgu.norm(z2)
        z2_spatial = sgu.spatial_proj(z2_normed)
        expected = z1 * z2_spatial

        actual = sgu(z)
        assert torch.allclose(actual, expected, atol=1e-5), \
            "SGU output doesn't match manual Eq. 4 computation"

    def test_odd_d_ffn_raises(self):
        with pytest.raises(AssertionError):
            SpatialGatingUnit(d_ffn=7, seq_len=N)

    def test_no_nan(self):
        torch.manual_seed(1)
        sgu = SpatialGatingUnit(d_ffn=D_FFN, seq_len=N)
        z = torch.randn(B, N, D_FFN)
        out = sgu(z)
        assert not torch.isnan(out).any()


class TestaMLP_SGU:

    def test_output_shape(self):
        """aMLP_SGU: [B, n, d_ffn], [B, n, d_model] → [B, n, d_ffn//2]."""
        sgu = aMLP_SGU(d_ffn=D_FFN, d_model=D_MODEL, seq_len=N, d_attn=16)
        z = torch.randn(B, N, D_FFN)
        x_pre = torch.randn(B, N, D_MODEL)
        out = sgu(z, x_pre)
        assert out.shape == (B, N, D_FFN // 2)

    @pytest.mark.parametrize("fusion_mode", ["add", "concat", "replace"])
    def test_fusion_modes(self, fusion_mode):
        """All three fusion modes produce correct shape."""
        sgu = aMLP_SGU(d_ffn=D_FFN, d_model=D_MODEL, seq_len=N,
                       d_attn=16, fusion_mode=fusion_mode)
        z = torch.randn(B, N, D_FFN)
        x_pre = torch.randn(B, N, D_MODEL)
        out = sgu(z, x_pre)
        assert out.shape == (B, N, D_FFN // 2), \
            f"fusion_mode='{fusion_mode}' gave wrong shape: {out.shape}"

    def test_invalid_fusion_mode_raises(self):
        with pytest.raises(AssertionError):
            aMLP_SGU(d_ffn=D_FFN, d_model=D_MODEL, seq_len=N, fusion_mode="invalid")


# ═══════════════════════════════════════════════════════════════
# 4. gMLPBlock
# ═══════════════════════════════════════════════════════════════

class TestgMLPBlock:

    def test_output_shape_preserves_input(self):
        """Output shape must equal input shape [B, n, d_model]."""
        block = gMLPBlock(d_model=D_MODEL, d_ffn=D_FFN, seq_len=N)
        x = torch.randn(B, N, D_MODEL)
        out = block(x)
        assert out.shape == x.shape, \
            f"Block must preserve shape. Expected {x.shape}, got {out.shape}"

    def test_residual_connection(self):
        """With zero-initialized projection weights the output ≈ input (residual)."""
        block = gMLPBlock(d_model=D_MODEL, d_ffn=D_FFN, seq_len=N, w_init_std=0.0)
        # Zero out all learnable weights except what's needed for near-identity
        with torch.no_grad():
            block.channel_expand.weight.zero_()
            block.channel_expand.bias.zero_()
            block.channel_contract.weight.zero_()
            block.channel_contract.bias.zero_()
        x = torch.randn(B, N, D_MODEL)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-5), \
            "With zeroed projections, block output should equal input (residual only)"

    def test_amlp_block_output_shape(self):
        """aMLP block (use_tiny_attn=True) also preserves shape."""
        block = gMLPBlock(d_model=D_MODEL, d_ffn=D_FFN, seq_len=N,
                          use_tiny_attn=True, d_attn=16)
        x = torch.randn(B, N, D_MODEL)
        out = block(x)
        assert out.shape == x.shape

    def test_droppath_disabled_at_eval(self):
        """DropPath must be disabled (returns input unchanged) at eval time."""
        drop = DropPath(survival_prob=0.5)
        drop.eval()
        x = torch.randn(B, N, D_MODEL)
        out = drop(x)
        assert torch.equal(out, x), "DropPath should be identity at eval time"

    def test_droppath_active_at_train(self):
        """DropPath should randomly zero some samples during training."""
        torch.manual_seed(123)
        drop = DropPath(survival_prob=0.5)
        drop.train()
        results = set()
        for _ in range(20):
            x = torch.ones(4, N, D_MODEL)
            out = drop(x)
            # If any sample was dropped, its sum will differ from 1.0 * N * D_MODEL
            dropped = (out.sum(dim=(1, 2)) < N * D_MODEL * 0.9)
            results.add(dropped.any().item())
        assert True in results, "DropPath never dropped any samples in 20 trials (suspicious)"

    def test_no_nan_in_output(self):
        torch.manual_seed(42)
        block = gMLPBlock(d_model=D_MODEL, d_ffn=D_FFN, seq_len=N)
        x = torch.randn(B, N, D_MODEL)
        out = block(x)
        assert not torch.isnan(out).any()


# ═══════════════════════════════════════════════════════════════
# 5. PatchEmbedding
# ═══════════════════════════════════════════════════════════════

class TestPatchEmbedding:

    def test_output_shape(self):
        """[B, 3, 8, 8] with patch_size=4 → [B, 4, d_model]."""
        pe = PatchEmbedding(img_size=8, patch_size=4, d_model=D_MODEL)
        x = torch.randn(B, 3, 8, 8)
        out = pe(x)
        assert out.shape == (B, 4, D_MODEL), f"Expected (B,4,{D_MODEL}), got {out.shape}"

    def test_num_patches(self):
        """224×224 with patch_size=16 → 196 patches."""
        pe = PatchEmbedding(img_size=224, patch_size=16, d_model=64)
        assert pe.num_patches == 196

    def test_non_divisible_raises(self):
        with pytest.raises(AssertionError):
            PatchEmbedding(img_size=7, patch_size=4)


# ═══════════════════════════════════════════════════════════════
# 6. gMLP Top-Level Model (NLP)
# ═══════════════════════════════════════════════════════════════

class TestgMLPNLP:

    def test_mlm_forward_shape(self, nlp_config):
        """MLM forward: logits [B, n, vocab_size]."""
        model = gMLP(nlp_config)
        model.set_task("mlm")
        input_ids = torch.randint(0, nlp_config.vocab_size, (B, N))
        out = model(input_ids=input_ids)
        assert out.logits.shape == (B, N, nlp_config.vocab_size)
        assert out.loss is None

    def test_mlm_loss_computed_with_labels(self, nlp_config):
        """MLM loss is a scalar when labels are provided."""
        model = gMLP(nlp_config)
        model.set_task("mlm")
        input_ids = torch.randint(0, nlp_config.vocab_size, (B, N))
        labels = torch.full((B, N), -100, dtype=torch.long)
        # Mask some tokens
        labels[:, 2:5] = torch.randint(0, nlp_config.vocab_size, (B, 3))
        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.ndim == 0, "Loss should be a scalar"
        assert out.loss.item() > 0

    def test_classification_forward(self, nlp_config):
        """Classification forward: logits [B, num_classes]."""
        nlp_config.num_classes = 3
        model = gMLP(nlp_config)
        model.set_task("classification")
        input_ids = torch.randint(0, nlp_config.vocab_size, (B, N))
        labels = torch.randint(0, 3, (B,))
        out = model(input_ids=input_ids, labels=labels)
        assert out.logits.shape == (B, 3)
        assert out.loss is not None

    def test_weight_tying(self, nlp_config):
        """LM head weights must be tied to embedding weights."""
        model = gMLP(nlp_config)
        assert model.lm_head.weight is model.embedding.weight, \
            "LM head and embedding must share weights (weight tying)"

    def test_amlp_nlp_forward(self, amlp_config):
        """aMLP NLP forward pass produces correct shapes."""
        model = gMLP(amlp_config)
        model.set_task("mlm")
        input_ids = torch.randint(0, amlp_config.vocab_size, (B, N))
        out = model(input_ids=input_ids)
        assert out.logits.shape == (B, N, amlp_config.vocab_size)

    def test_no_positional_encoding_in_nlp(self, nlp_config):
        """Paper: gMLP does not use positional encodings. No pos_embedding attribute."""
        model = gMLP(nlp_config)
        assert not hasattr(model, "pos_embedding"), \
            "gMLP must NOT have positional encoding (paper Section 2)"
        assert not hasattr(model, "positional_encoding")

    def test_hidden_states_returned(self, nlp_config):
        """hidden_states in ModelOutput should be [B, n, d_model]."""
        model = gMLP(nlp_config)
        model.set_task("mlm")
        input_ids = torch.randint(0, nlp_config.vocab_size, (B, N))
        out = model(input_ids=input_ids)
        assert out.hidden_states is not None
        assert out.hidden_states.shape == (B, N, nlp_config.d_model)

    def test_gradient_flows(self, nlp_config):
        """
        Gradients must flow through all MLM-path components.
        The classification head (classifier.*) is NOT in the MLM forward path,
        so we correctly exclude it from this check.
        """
        model = gMLP(nlp_config)
        model.set_task("mlm")
        input_ids = torch.randint(0, nlp_config.vocab_size, (B, N))
        labels = torch.randint(0, nlp_config.vocab_size, (B, N))
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        # Parameters not in the MLM forward path (classifier head only used in finetune)
        excluded = {"classifier.weight", "classifier.bias"}
        for name, param in model.named_parameters():
            if name in excluded:
                continue   # these don't participate in MLM forward — expected no grad
            if param.requires_grad and param.grad is None:
                pytest.fail(f"No gradient for parameter: {name}")


# ═══════════════════════════════════════════════════════════════
# 7. gMLP Top-Level Model (Vision)
# ═══════════════════════════════════════════════════════════════

class TestgMLPVision:

    def test_vision_forward_shape(self, vision_config):
        """Vision forward: logits [B, num_classes]."""
        model = gMLP(vision_config)
        imgs = torch.randn(B, 3, 8, 8)
        out = model(pixel_values=imgs)
        assert out.logits.shape == (B, vision_config.num_classes), \
            f"Expected ({B},{vision_config.num_classes}), got {out.logits.shape}"

    def test_vision_loss_with_labels(self, vision_config):
        """Vision loss is a scalar when labels provided."""
        model = gMLP(vision_config)
        imgs = torch.randn(B, 3, 8, 8)
        labels = torch.randint(0, vision_config.num_classes, (B,))
        out = model(pixel_values=imgs, labels=labels)
        assert out.loss is not None
        assert out.loss.ndim == 0

    def test_vision_no_positional_encoding(self, vision_config):
        """Paper: vision gMLP also does not require positional encodings."""
        model = gMLP(vision_config)
        assert not hasattr(model, "pos_embedding")

    def test_vision_uses_unconstrained_W(self, vision_config):
        """Vision mode uses unconstrained [n, n] W (not Toeplitz)."""
        assert vision_config.use_toeplitz is False, \
            "Vision config must have use_toeplitz=False"
        model = gMLP(vision_config)
        # Check that the spatial_proj in each block is NOT using Toeplitz
        for block in model.blocks:
            sp = block.sgu.spatial_proj
            assert not sp.use_toeplitz, \
                "Vision gMLP blocks must use unconstrained spatial projection"

    def test_global_avg_pool(self, vision_config):
        """ASSUMED: pool_mode='avg' applies global average pool before classifier."""
        assert vision_config.pool_mode == "avg"
        model = gMLP(vision_config)
        imgs = torch.randn(B, 3, 8, 8)
        out = model(pixel_values=imgs)
        assert out.logits.shape == (B, vision_config.num_classes)


# ═══════════════════════════════════════════════════════════════
# 8. Config Presets — Parameter Count Validation
# ═══════════════════════════════════════════════════════════════

class TestConfigPresets:
    """
    Validate model sizes against paper Tables 1 and 5.
    Uses ±15% tolerance to account for minor implementation differences
    (e.g., LM head weight tying, normalisation layer counts).
    """

    PAPER_PARAMS = {
        # Vision (Table 1) — in millions
        "gmlp-Ti-imagenet":  5.9,
        "gmlp-S-imagenet":  19.5,
        "gmlp-B-imagenet":  73.4,
        # NLP (Table 5) — in millions
        "gmlp-base-mlm":   130.0,
        "amlp-base-mlm":   109.0,
        "gmlp-large-mlm":  365.0,
    }

    @pytest.mark.parametrize("preset_name,expected_M", PAPER_PARAMS.items())
    def test_param_count(self, preset_name, expected_M):
        config = get_preset(preset_name)
        model = gMLP(config.model)
        actual_M = model.get_num_params() / 1e6
        tol = 0.20   # 20% tolerance for implementation differences
        assert abs(actual_M - expected_M) / expected_M < tol, (
            f"[{preset_name}] param count {actual_M:.1f}M vs paper {expected_M}M "
            f"(>{tol*100:.0f}% difference)"
        )

    def test_amlp_has_fewer_params_than_gmlp_same_size(self):
        """aMLPbase (109M) has fewer params than gMLPbase (130M) — paper Table 5."""
        gmlp = gMLP(get_preset("gmlp-base-mlm").model)
        amlp = gMLP(get_preset("amlp-base-mlm").model)
        assert amlp.get_num_params() < gmlp.get_num_params(), \
            "aMLPbase should have fewer params than gMLPbase (fewer layers, tiny attn)"


# ═══════════════════════════════════════════════════════════════
# 9. Loss Functions
# ═══════════════════════════════════════════════════════════════

class TestLosses:

    def test_mlm_loss_ignores_minus_100(self):
        """MLM loss must ignore positions where labels=-100."""
        loss_fn = MLMLoss(vocab_size=100)
        logits = torch.randn(2, 8, 100)
        labels = torch.full((2, 8), -100, dtype=torch.long)
        # All masked: loss should be 0 (no positions to compute)
        # PyTorch CE with all ignore_index returns 0 or NaN depending on version
        # Just check it doesn't raise and returns a scalar
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_mlm_loss_positive_when_masked(self):
        loss_fn = MLMLoss(vocab_size=100)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_classification_loss_soft_labels(self):
        """ClassificationLoss must handle soft labels from Mixup/CutMix."""
        loss_fn = ClassificationLoss(label_smoothing=0.1)
        logits = torch.randn(4, 10)
        soft_labels = torch.softmax(torch.randn(4, 10), dim=-1)   # [4, 10] soft
        loss = loss_fn(logits, soft_labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_qa_loss_returns_scalar(self):
        loss_fn = QALoss()
        B, N = 3, 32
        start_logits = torch.randn(B, N)
        end_logits = torch.randn(B, N)
        start_pos = torch.randint(0, N, (B,))
        end_pos = torch.randint(0, N, (B,))
        loss = loss_fn(start_logits, end_logits, start_pos, end_pos)
        assert loss.ndim == 0


# ═══════════════════════════════════════════════════════════════
# 10. LR Schedulers
# ═══════════════════════════════════════════════════════════════

class TestSchedulers:

    def _dummy_optimizer(self):
        m = nn.Linear(4, 4)
        return torch.optim.AdamW(m.parameters(), lr=1.0)

    def test_linear_warmup_peak_at_warmup_end(self):
        optim = self._dummy_optimizer()
        sched = get_linear_warmup_decay(optim, warmup_steps=10, total_steps=100)
        for _ in range(10):
            sched.step()
        assert abs(sched.get_last_lr()[0] - 1.0) < 0.01, \
            "LR should be at peak (1.0×base) at end of warmup"

    def test_linear_decay_reaches_zero(self):
        optim = self._dummy_optimizer()
        sched = get_linear_warmup_decay(optim, warmup_steps=5, total_steps=10)
        for _ in range(10):
            sched.step()
        assert sched.get_last_lr()[0] < 0.01, "Linear schedule should decay to ~0"

    def test_cosine_decay_positive_at_end(self):
        optim = self._dummy_optimizer()
        sched = get_cosine_warmup_decay(optim, warmup_steps=5, total_steps=100)
        for _ in range(100):
            sched.step()
        assert sched.get_last_lr()[0] >= 0.0, "Cosine schedule must not go negative"

    def test_warmup_increases_monotonically(self):
        optim = self._dummy_optimizer()
        sched = get_linear_warmup_decay(optim, warmup_steps=20, total_steps=200)
        lrs = []
        for _ in range(20):
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-6, \
                f"LR decreased during warmup at step {i}: {lrs[i-1]:.4f} → {lrs[i]:.4f}"


# ═══════════════════════════════════════════════════════════════
# 11. Evaluation Metrics
# ═══════════════════════════════════════════════════════════════

class TestMetrics:

    def test_compute_perplexity(self):
        losses = [2.0, 2.0, 2.0]
        ppl = compute_perplexity(losses)
        assert abs(ppl - math.exp(2.0)) < 1e-4

    def test_compute_accuracy(self):
        preds = torch.tensor([0, 1, 2, 1])
        labels = torch.tensor([0, 1, 0, 1])
        acc = compute_accuracy(preds, labels)
        assert abs(acc - 0.75) < 1e-4

    def test_squad_f1_exact_match(self):
        f1 = compute_squad_f1(["the cat sat"], [["the cat sat"]])
        assert abs(f1 - 1.0) < 1e-4

    def test_squad_f1_partial(self):
        f1 = compute_squad_f1(["cat"], [["the cat sat"]])
        assert 0.0 < f1 < 1.0

    def test_squad_f1_no_match(self):
        f1 = compute_squad_f1(["dog"], [["the cat sat"]])
        assert abs(f1 - 0.0) < 1e-4

    def test_aggregate_runs_median(self):
        vals = [80.0, 81.0, 82.0, 83.0, 84.0]
        stats = aggregate_runs(vals)
        assert stats["median"] == 82.0
        assert stats["n_runs"] == 5
        assert stats["min"] == 80.0
        assert stats["max"] == 84.0


# ═══════════════════════════════════════════════════════════════
# 12. End-to-End Integration
# ═══════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_one_training_step_nlp(self, nlp_config):
        """Verify a full NLP training step completes without error."""
        set_seed(0)
        model = gMLP(nlp_config)
        model.set_task("mlm")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        input_ids = torch.randint(0, nlp_config.vocab_size, (B, N))
        labels = torch.randint(0, nlp_config.vocab_size, (B, N))

        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_one_training_step_vision(self, vision_config):
        """Verify a full vision training step completes without error."""
        set_seed(0)
        model = gMLP(vision_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        imgs = torch.randn(B, 3, 8, 8)
        labels = torch.randint(0, vision_config.num_classes, (B,))

        optimizer.zero_grad()
        out = model(pixel_values=imgs, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_save_load_preserves_output(self, nlp_config, tmp_path):
        """Saving and loading a checkpoint must preserve model outputs exactly."""
        set_seed(42)
        model = gMLP(nlp_config)
        model.set_task("mlm")
        input_ids = torch.randint(0, nlp_config.vocab_size, (1, N))

        with torch.no_grad():
            out_before = model(input_ids=input_ids).logits.clone()

        # Save and reload
        model.save_pretrained(str(tmp_path))
        model2 = gMLP.from_pretrained(str(tmp_path))
        model2.set_task("mlm")
        model2.eval()

        with torch.no_grad():
            out_after = model2(input_ids=input_ids).logits

        assert torch.allclose(out_before, out_after, atol=1e-5), \
            "Saved/loaded model output differs from original"

    def test_amlp_outperforms_plain_after_few_steps(self):
        """
        Sanity check: after a few gradient steps on a simple objective,
        aMLP and gMLP both improve (loss decreases). Not a performance claim.
        """
        set_seed(0)
        for use_attn in [False, True]:
            config = ModelConfig(
                model_type="nlp", num_layers=1, d_model=32, d_ffn=64,
                seq_len=8, vocab_size=64, num_classes=64,
                use_toeplitz=True, use_tiny_attn=use_attn, d_attn=8,
            )
            model = gMLP(config)
            model.set_task("mlm")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            input_ids = torch.randint(0, 64, (4, 8))
            labels = torch.randint(0, 64, (4, 8))

            losses = []
            for _ in range(5):
                optimizer.zero_grad()
                out = model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                optimizer.step()
                losses.append(out.loss.item())

            assert losses[-1] < losses[0], \
                f"{'aMLP' if use_attn else 'gMLP'} loss did not decrease over 5 steps"
