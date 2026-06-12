"""
gmlp/models/gmlp.py
-------------------
Top-level gMLP model supporting NLP (MLM / finetuning) and vision (ImageNet).

Paper: "Pay Attention to MLPs" — Liu et al., 2021 (arXiv:2105.08050)

Architecture overview:
  NLP:    Embedding → L × gMLPBlock → LayerNorm → LM/Classification head
  Vision: PatchEmbed → L × gMLPBlock → LayerNorm → AvgPool → Classifier

Input/output protocols strictly follow BERT (NLP) and ViT/B16 (vision)
as stated in Section 2 and Figure 1. No positional encodings are used.

Key design difference from Transformer:
  - Self-attention replaced by Spatial Gating Unit (W ∈ R^{n×n}, static)
  - Same W shared across all channels (not per-head, not input-dependent)
  - gMLP scales with compute/data comparably to BERT (Section 4.2)

Paper ref: Sections 2–4, Tables 1, 5, 6
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from .gmlp_block import gMLPBlock
from .patch_embed import PatchEmbedding
from ..utils.config import ModelConfig


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    """Container for gMLP forward pass outputs."""
    logits: Tensor                           # [B, n, vocab] for MLM; [B, C] for clf
    loss: Optional[Tensor] = None            # scalar if labels provided
    hidden_states: Optional[Tensor] = None  # [B, n, d_model] last layer


# ---------------------------------------------------------------------------
# gMLP model
# ---------------------------------------------------------------------------

class gMLP(nn.Module):
    """
    Unified gMLP / aMLP model for NLP and vision.

    Dispatches to NLP or vision mode based on config.model_type.

    NLP mode (BERT-style):
      - Token embedding table (no positional encoding)
      - L stacked gMLPBlocks with Toeplitz W
      - Final LayerNorm
      - MLM head (weight-tied to embedding) or classification head

    Vision mode (ViT-style):
      - 16×16 patch embedding (Conv2d stem)
      - L stacked gMLPBlocks with unconstrained W
      - Final LayerNorm
      - Global average pool → linear classifier
      # ASSUMED: global avg pool (SIR ambiguity_003, conf=0.70)
      # Config flag pool_mode='cls' available as alternative

    Args:
        config: ModelConfig dataclass with all architecture hyperparameters.

    Paper ref: Sections 2–4
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.model_type = config.model_type

        # ---------------------------------------------------------------
        # Input layer
        # ---------------------------------------------------------------
        if config.model_type == "nlp":
            # Token embedding: vocab_size × d_model
            # No positional embedding (paper Section 2, Figure 1 caption)
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.input_norm = None   # pre-norm handled inside each block
        else:
            # Vision: ViT-style patch embedding
            self.patch_embed = PatchEmbedding(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_channels=3,
                d_model=config.d_model,
            )
            # Update seq_len from patch grid (overrides config if needed)
            actual_seq = self.patch_embed.num_patches
            if actual_seq != config.seq_len:
                import warnings
                warnings.warn(
                    f"config.seq_len={config.seq_len} but patch grid gives "
                    f"{actual_seq} patches; using {actual_seq}."
                )
                config.seq_len = actual_seq

        # ---------------------------------------------------------------
        # Stack of L identical gMLPBlocks
        # ---------------------------------------------------------------
        self.blocks = nn.ModuleList([
            gMLPBlock(
                d_model=config.d_model,
                d_ffn=config.d_ffn,
                seq_len=config.seq_len,
                use_toeplitz=config.use_toeplitz,
                use_tiny_attn=config.use_tiny_attn,
                d_attn=config.d_attn,
                w_init_std=config.w_init_std,
                attn_fusion_mode=config.attn_fusion_mode,
                survival_prob=config.survival_prob,
            )
            for _ in range(config.num_layers)
        ])

        # Final LayerNorm (after all blocks, before head)
        self.final_norm = nn.LayerNorm(config.d_model)

        # ---------------------------------------------------------------
        # Output head
        # ---------------------------------------------------------------
        if config.model_type == "nlp":
            # MLM head: d_model → vocab_size (weight-tied to embedding)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            # Weight tying: share parameters with embedding matrix
            self.lm_head.weight = self.embedding.weight

            # Classification head for finetuning (CLS token → num_classes)
            # Initialized separately; only active in finetune mode
            self.classifier = nn.Linear(config.d_model, config.num_classes)

        else:  # vision
            # Global average pool then classifier
            self.classifier = nn.Linear(config.d_model, config.num_classes)

        # Mode: 'mlm', 'classification', 'qa' — set externally for finetuning
        self._task = "mlm" if config.model_type == "nlp" else "classification"

        self._init_weights()

    def _init_weights(self) -> None:
        """Standard initialisation for linear and embedding layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Note: ToeplitzLinear weights are initialised inside ToeplitzLinear.__init__
        # with near-zero std (w_init_std) and bias=1. _init_weights() must NOT
        # overwrite them — ToeplitzLinear does not subclass nn.Linear.

    def set_task(self, task: str) -> None:
        """Switch between 'mlm', 'classification', 'qa' heads."""
        assert task in ("mlm", "classification", "qa"), f"Unknown task: {task}"
        self._task = task

    def get_num_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[Tensor] = None,       # [B, n] int64  (NLP)
        pixel_values: Optional[Tensor] = None,    # [B, 3, H, W]  (Vision)
        labels: Optional[Tensor] = None,          # [B, n] for MLM; [B] for clf
        attention_mask: Optional[Tensor] = None,  # [B, n] (used for loss masking only)
    ) -> ModelOutput:
        """
        Unified forward pass.

        For NLP: pass input_ids (and optionally labels for loss computation).
        For Vision: pass pixel_values (and optionally labels).

        Paper note: attention_mask is NOT used inside gMLP blocks —
        the model naturally ignores padding tokens (Section 4, Figure 1 caption).
        It is only used to mask out padding positions during loss computation.

        Returns:
            ModelOutput with logits, loss (if labels given), hidden_states.
        """
        if self.model_type == "nlp":
            return self._forward_nlp(input_ids, labels, attention_mask)
        else:
            return self._forward_vision(pixel_values, labels)

    def _forward_nlp(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> ModelOutput:
        """NLP forward: embedding → blocks → norm → head."""
        assert input_ids is not None, "input_ids required for NLP mode"
        assert input_ids.dim() == 2, f"Expected [B, n], got {input_ids.shape}"

        # Token embedding (no positional encoding)
        x = self.embedding(input_ids)          # [B, n, d_model]

        # Pass through L gMLPBlocks
        for block in self.blocks:
            x = block(x)                       # [B, n, d_model]

        # Final LayerNorm
        x = self.final_norm(x)                 # [B, n, d_model]
        hidden_states = x

        # ── Head dispatch ──
        if self._task == "mlm":
            # Weight-tied LM head: [B, n, vocab_size]
            logits = self.lm_head(x)
            loss = None
            if labels is not None:
                # Compute MLM loss only on masked positions
                # attention_mask used to exclude padding from loss
                loss = self._mlm_loss(logits, labels, attention_mask)
            return ModelOutput(logits=logits, loss=loss, hidden_states=hidden_states)

        elif self._task == "classification":
            # Use CLS token (position 0) for classification
            # Paper protocol: "predictions deduced from last-layer representation
            # of a reserved <cls> symbol" (Section 2)
            cls_repr = x[:, 0, :]             # [B, d_model]
            logits = self.classifier(cls_repr) # [B, num_classes]
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
            return ModelOutput(logits=logits, loss=loss, hidden_states=hidden_states)

        else:
            raise ValueError(f"Unknown task for NLP: {self._task}")

    def _forward_vision(
        self,
        pixel_values: Tensor,
        labels: Optional[Tensor] = None,
    ) -> ModelOutput:
        """Vision forward: patch embed → blocks → norm → pool → classifier."""
        assert pixel_values is not None, "pixel_values required for vision mode"
        assert pixel_values.dim() == 4, f"Expected [B, 3, H, W], got {pixel_values.shape}"

        # Patch embedding: [B, 3, H, W] → [B, num_patches, d_model]
        x = self.patch_embed(pixel_values)

        # Pass through L gMLPBlocks
        for block in self.blocks:
            x = block(x)                       # [B, num_patches, d_model]

        # Final LayerNorm
        x = self.final_norm(x)                 # [B, num_patches, d_model]
        hidden_states = x

        # Pooling
        # ASSUMED: global average pool (SIR ambiguity_003, conf=0.70)
        # config.pool_mode='cls' uses x[:,0,:] instead
        if self.config.pool_mode == "avg":
            pooled = x.mean(dim=1)             # [B, d_model]
        else:  # 'cls'
            pooled = x[:, 0, :]               # [B, d_model]

        logits = self.classifier(pooled)       # [B, num_classes]
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits, labels,
                label_smoothing=0.1,           # paper Table 7: label_smoothing=0.1
            )
        return ModelOutput(logits=logits, loss=loss, hidden_states=hidden_states)

    def _mlm_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Cross-entropy loss over masked token positions only.
        labels = -100 at non-masked / padding positions (standard HuggingFace convention).
        """
        # logits: [B, n, vocab_size]  labels: [B, n]
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),  # [B*n, vocab]
            labels.view(-1),                   # [B*n]
            ignore_index=-100,                 # -100 = not a masked token
        )
        return loss

    @classmethod
    def from_pretrained(cls, path: str, config: Optional[ModelConfig] = None) -> "gMLP":
        """Load a saved checkpoint."""
        import os
        ckpt = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        if config is None:
            from ..utils.config import ModelConfig
            config = ModelConfig(**ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def save_pretrained(self, path: str) -> None:
        """Save model weights + config."""
        import os
        from dataclasses import asdict
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
        }, os.path.join(path, "model.pt"))

    def __repr__(self) -> str:
        return (
            f"gMLP(model_type={self.model_type}, "
            f"L={len(self.blocks)}, d_model={self.config.d_model}, "
            f"d_ffn={self.config.d_ffn}, "
            f"use_tiny_attn={self.config.use_tiny_attn}, "
            f"params={self.get_num_params():,})"
        )
