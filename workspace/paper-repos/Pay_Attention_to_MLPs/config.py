"""
gmlp/utils/config.py
--------------------
Central configuration dataclass for the gMLP / aMLP architecture.

Implements the config schema from Stage 3 architecture plan.
All hyperparameters are sourced from the SIR (sir.json, arxiv_2105_08050).
Fields annotated with # ASSUMED indicate values not explicitly stated in the paper.

Paper: "Pay Attention to MLPs" — Liu et al., 2021 (arXiv:2105.08050)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Architecture hyperparameters for gMLP / aMLP.

    Args:
        model_type:      'nlp' or 'vision'. Determines input protocol.
        use_tiny_attn:   If True, builds aMLP (gMLP + tiny single-head attention).
        num_layers:      Number of stacked gMLPBlock layers (L in paper).
        d_model:         Token hidden dimension.
        d_ffn:           FFN expansion dimension (≈4–6× d_model).
        seq_len:         Maximum sequence length (n). Determines W shape.
        vocab_size:      Vocabulary size (NLP only). Paper: 32K SentencePiece cased.
        num_classes:     Output classes (1000 for ImageNet, vocab_size for MLM).
        img_size:        Input image resolution (vision only). Paper: 224.
        patch_size:      ViT patch size (vision only). Paper: 16.
        use_toeplitz:    Constrain W to Toeplitz matrix. True for NLP, False for vision.
        d_attn:          Tiny-attention hidden dim (aMLP only). Paper: 64 or 128.
        w_init_std:      Std for near-zero W init. Paper says "near-zero"; not specified.
                         # ASSUMED: 0.002 (10× smaller than standard 0.02 init)
        attn_fusion_mode: How tiny_attn output combines with spatial gate.
                         # ASSUMED: 'add' (additive, from Fig.6 diagram)
                         # TODO: verify from paper — ambiguity_001
        pool_mode:       Vision classification pooling: 'avg' or 'cls'.
                         # ASSUMED: 'avg' (global average pool) — ambiguity_003
        survival_prob:   Stochastic depth survival probability (vision only).
                         Paper Table 1: Ti=1.0, S=0.95, B=0.80.
    """
    model_type: str = "nlp"
    use_tiny_attn: bool = False
    num_layers: int = 48
    d_model: int = 512
    d_ffn: int = 3072
    seq_len: int = 512
    vocab_size: int = 32000
    num_classes: int = 32000
    img_size: int = 224
    patch_size: int = 16
    use_toeplitz: bool = True
    d_attn: int = 64
    # ASSUMED: near-zero init std not specified in paper (SIR ambiguity_002, conf=0.65)
    w_init_std: float = 0.002
    # ASSUMED: additive fusion per Fig.6 (SIR ambiguity_001, conf=0.75) TODO:verify
    attn_fusion_mode: str = "add"
    # ASSUMED: global avg pool (SIR ambiguity_003, conf=0.70)
    pool_mode: str = "avg"
    survival_prob: float = 1.0

    def validate(self) -> None:
        if self.model_type not in ("nlp", "vision"):
            raise ValueError(f"model_type must be 'nlp' or 'vision', got '{self.model_type}'")
        if self.d_ffn % 2 != 0:
            raise ValueError(f"d_ffn must be even (SGU splits it in half), got {self.d_ffn}")
        if self.use_tiny_attn and self.attn_fusion_mode not in ("add", "concat", "replace"):
            raise ValueError(f"attn_fusion_mode must be 'add'/'concat'/'replace', got '{self.attn_fusion_mode}'")
        if self.pool_mode not in ("avg", "cls"):
            raise ValueError(f"pool_mode must be 'avg' or 'cls', got '{self.pool_mode}'")
        if not (0.0 < self.survival_prob <= 1.0):
            raise ValueError(f"survival_prob must be in (0, 1], got {self.survival_prob}")


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Optimizer, LR schedule, and training loop settings.

    All values from Table 7 (vision) and Table 8 (NLP) of the paper.
    """
    optimizer: str = "adamw"
    lr: float = 1e-4                     # paper Table 8: 1e-4 (NLP full), 7e-4 (ablation)
    weight_decay: float = 0.01           # paper Table 8: 0.01 (NLP), 0.05 (vision)
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-6
    grad_clip: float = 0.0               # paper: 0 for NLP, 1.0 for vision
    lr_schedule: str = "linear"          # 'linear' (NLP) or 'cosine' (vision)
    warmup_steps: int = 10000
    batch_size: int = 256
    num_steps: int = 1_000_000           # paper: 1M steps (full BERT setup)
    num_epochs: int = 300                # vision only
    log_interval: int = 100
    save_interval: int = 10_000
    eval_interval: int = 10_000
    precision: str = "float32"           # 'float32' or 'bf16'
    num_workers: int = 4
    seed: int = 42
    deterministic: bool = False          # note: setting True may slow training significantly

    def validate(self) -> None:
        if self.optimizer not in ("adamw", "adam", "sgd"):
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.lr_schedule not in ("linear", "cosine", "constant"):
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
        if self.precision not in ("float32", "bf16", "fp16"):
            raise ValueError(f"Unknown precision: {self.precision}")


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset paths and preprocessing settings."""
    dataset: str = "c4"                  # 'c4', 'imagenet', 'glue', 'squad'
    data_dir: str = "data/"
    mlm_probability: float = 0.15       # BERT-style MLM masking probability
    max_seq_len: int = 512
    # Vision augmentation (paper Table 7)
    autoaugment: bool = True
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    cutmix_mixup_switch_prob: float = 0.5
    label_smoothing: float = 0.1
    repeated_augmentation: bool = False  # paper: off (unlike DeiT)
    random_erasing_prob: float = 0.0    # paper: 0
    use_streaming: bool = True          # recommended for C4 (~300GB)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class gMLPConfig:
    """
    Top-level configuration container for gMLP experiments.
    Serializes to/from YAML.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "outputs/"
    experiment_name: str = "gmlp_base_mlm"

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()

    @classmethod
    def from_yaml(cls, path: str) -> "gMLPConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            output_dir=raw.get("output_dir", "outputs/"),
            experiment_name=raw.get("experiment_name", "gmlp"),
        )
        cfg.validate()
        return cfg

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return (
            f"gMLPConfig(model_type={self.model.model_type}, "
            f"L={self.model.num_layers}, d_model={self.model.d_model}, "
            f"d_ffn={self.model.d_ffn}, use_tiny_attn={self.model.use_tiny_attn})"
        )


# ---------------------------------------------------------------------------
# Preset factories (paper Table 1 & 5 values)
# ---------------------------------------------------------------------------

PRESETS = {
    # NLP (full BERT setup, paper Table 5)
    "gmlp-base-mlm": dict(
        model=dict(model_type="nlp", use_tiny_attn=False, num_layers=48,
                   d_model=512, d_ffn=3072, seq_len=512, use_toeplitz=True),
        training=dict(lr=1e-4, weight_decay=0.01, batch_size=256, num_steps=1_000_000,
                      lr_schedule="linear", grad_clip=0.0),
    ),
    "amlp-base-mlm": dict(
        model=dict(model_type="nlp", use_tiny_attn=True, num_layers=36,
                   d_model=512, d_ffn=3072, seq_len=512, d_attn=64, use_toeplitz=True),
        training=dict(lr=1e-4, weight_decay=0.01, batch_size=256, num_steps=1_000_000,
                      lr_schedule="linear", grad_clip=0.0),
    ),
    "gmlp-large-mlm": dict(
        model=dict(model_type="nlp", use_tiny_attn=False, num_layers=96,
                   d_model=768, d_ffn=3072, seq_len=512, use_toeplitz=True),
        training=dict(lr=1e-4, weight_decay=0.01, batch_size=256, num_steps=1_000_000,
                      lr_schedule="linear", grad_clip=0.0),
    ),
    "amlp-large-mlm": dict(
        model=dict(model_type="nlp", use_tiny_attn=True, num_layers=72,
                   d_model=768, d_ffn=3072, seq_len=512, d_attn=128, use_toeplitz=True),
        training=dict(lr=1e-4, weight_decay=0.01, batch_size=256, num_steps=1_000_000,
                      lr_schedule="linear", grad_clip=0.0),
    ),
    # Ablation (faster; paper Section 4.1)
    "gmlp-base-ablation": dict(
        model=dict(model_type="nlp", use_tiny_attn=False, num_layers=36,
                   d_model=512, d_ffn=3072, seq_len=128, use_toeplitz=True),
        training=dict(lr=7e-4, weight_decay=0.01, batch_size=2048, num_steps=125_000,
                      lr_schedule="linear", grad_clip=0.0),
        data=dict(dataset="c4", max_seq_len=128),
    ),
    # Vision (paper Table 1)
    "gmlp-Ti-imagenet": dict(
        model=dict(model_type="vision", use_tiny_attn=False, num_layers=30,
                   d_model=128, d_ffn=768, seq_len=196, num_classes=1000,
                   use_toeplitz=False, survival_prob=1.00),
        training=dict(lr=1e-3, weight_decay=0.05, batch_size=4096, num_epochs=300,
                      lr_schedule="cosine", grad_clip=1.0, warmup_steps=10000),
        data=dict(dataset="imagenet"),
    ),
    "gmlp-S-imagenet": dict(
        model=dict(model_type="vision", use_tiny_attn=False, num_layers=30,
                   d_model=256, d_ffn=1536, seq_len=196, num_classes=1000,
                   use_toeplitz=False, survival_prob=0.95),
        training=dict(lr=1e-3, weight_decay=0.05, batch_size=4096, num_epochs=300,
                      lr_schedule="cosine", grad_clip=1.0, warmup_steps=10000),
        data=dict(dataset="imagenet"),
    ),
    "gmlp-B-imagenet": dict(
        model=dict(model_type="vision", use_tiny_attn=False, num_layers=30,
                   d_model=512, d_ffn=3072, seq_len=196, num_classes=1000,
                   use_toeplitz=False, survival_prob=0.80),
        training=dict(lr=1e-3, weight_decay=0.05, batch_size=4096, num_epochs=300,
                      lr_schedule="cosine", grad_clip=1.0, warmup_steps=10000),
        data=dict(dataset="imagenet"),
    ),
}


def get_preset(name: str) -> gMLPConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    p = PRESETS[name]
    cfg = gMLPConfig(
        model=ModelConfig(**p.get("model", {})),
        training=TrainingConfig(**p.get("training", {})),
        data=DataConfig(**p.get("data", {})),
        experiment_name=name,
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    Called at the start of every training entrypoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # NOTE: deterministic mode may significantly slow down training
    except ImportError:
        pass
