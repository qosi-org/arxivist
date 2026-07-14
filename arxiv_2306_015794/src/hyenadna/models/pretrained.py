"""Load official pretrained HyenaDNA backbone + classification head.

Reproduction *critical path*. Builds the authors' exact architecture
(`standalone_hyenadna.HyenaDNAModel`) from the checkpoint's `config.json`, then
loads the released `weights.ckpt` state dict key-for-key. This is what makes the
reproduction faithful — a hand-written model cannot match the checkpoint keys.

Weights come from the HuggingFace Hub (e.g. `LongSafari/hyenadna-tiny-1k-seqlen`).
The `.ckpt` is a Lightning checkpoint whose keys are prefixed `model.`; we strip
that prefix and load into `HyenaDNAModel` (use_head=False) so the backbone returns
hidden states, and we attach our own classification head.
"""
from __future__ import annotations

import json
from typing import Optional

import torch
import torch.nn as nn

from .standalone_hyenadna import HyenaDNAModel

_HF_ORG = "LongSafari"
_WEIGHT_FILES = ["weights.ckpt", "weight.ckpt", "pytorch_model.bin"]


class HyenaDNAClassifier(nn.Module):
    """Official HyenaDNA backbone with a linear classification head.

    Args:
        backbone: HyenaDNAModel (use_head=False) returning [B, L, D] hidden states.
        d_model: backbone hidden size.
        num_classes: number of output classes.
        pool: 'mean' or 'last' pooling over the sequence dimension.
    """

    def __init__(self, backbone: nn.Module, d_model: int, num_classes: int, pool: str = "mean") -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.classifier = nn.Linear(d_model, num_classes)

    def __repr__(self) -> str:  # noqa: D105
        return f"HyenaDNAClassifier(pool={self.pool}, head={self.classifier})"

    @classmethod
    def from_pretrained(
        cls,
        variant: str,
        num_classes: int,
        d_model: int,
        pool: str = "mean",
        device: str = "cpu",
    ) -> "HyenaDNAClassifier":
        repo_id = variant if "/" in variant else f"{_HF_ORG}/{variant}"
        backbone, hidden = _build_and_load(repo_id, d_model)
        model = cls(backbone, d_model=hidden, num_classes=num_classes, pool=pool)
        return model.to(device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert input_ids.dim() == 2, f"Expected [B, L], got {tuple(input_ids.shape)}"
        hidden = self.backbone(input_ids)  # [B, L, D]
        if self.pool == "mean":
            pooled = hidden.mean(dim=1)
        elif self.pool == "last":
            pooled = hidden[:, -1, :]
        else:
            raise ValueError(f"Unknown pool={self.pool!r}")
        return self.classifier(pooled)


def _build_and_load(repo_id: str, d_model_default: int) -> tuple[nn.Module, int]:
    """Download config + weights, build HyenaDNAModel, load state dict.

    On any failure prints a clear warning and returns an untrained backbone.
    """
    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_id, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        d_model = int(cfg.get("d_model", d_model_default))
        n_layer = int(cfg.get("n_layer", 2))
        d_inner = int(cfg.get("d_inner", 4 * d_model))
        vocab_size = int(cfg.get("vocab_size", 12))
        # vocab is padded up to this multiple (checkpoint embeddings reflect the
        # padded size, e.g. 12 -> 16 with multiple=8).
        pad_mult = int(cfg.get("pad_vocab_size_multiple", 1))
        residual_in_fp32 = bool(cfg.get("residual_in_fp32", False))
        # Full Hyena mixer config from the checkpoint (emb_dim, filter_order,
        # l_max, w, modulate, ...); strip only the registry name key.
        layer = dict(cfg.get("layer", {"emb_dim": 5, "filter_order": 64, "l_max": 1026}))
        layer.pop("_name_", None)
        layer.setdefault("l_max", int(cfg.get("l_max", 1026)))

        backbone = HyenaDNAModel(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            layer=layer, use_head=False, pad_vocab_size_multiple=pad_mult,
            residual_in_fp32=residual_in_fp32,
        )

        state = _download_state_dict(repo_id)
        if state is None:
            raise FileNotFoundError(f"no weight file among {_WEIGHT_FILES}")

        # Lightning prefixes backbone params with 'model.'; strip it. Drop the
        # decoder/head keys — we use our own classification head.
        clean = {}
        for k, v in state.items():
            nk = k[len("model."):] if k.startswith("model.") else k
            clean[nk] = v
        missing, unexpected = backbone.load_state_dict(clean, strict=False)
        loaded = len(backbone.state_dict()) - len([m for m in missing])
        print(f"[pretrained] built HyenaDNAModel(d_model={d_model}, n_layer={n_layer}) "
              f"and loaded weights from {repo_id} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if len(missing) > len(backbone.state_dict()) // 2:
            raise RuntimeError(f"too many missing keys ({len(missing)}) — architecture mismatch")
        return backbone, d_model

    except Exception as exc:  # noqa: BLE001
        print(
            f"[WARNING] Could not load pretrained weights from '{repo_id}': {exc}\n"
            f"          Falling back to an untrained HyenaDNAModel. "
            f"Results will NOT match the paper."
        )
        layer = {"emb_dim": 5, "filter_order": 64, "short_filter_order": 3, "l_max": 1026}
        backbone = HyenaDNAModel(d_model=d_model_default, n_layer=2, d_inner=4 * d_model_default,
                                 vocab_size=12, layer=layer, use_head=False)
        return backbone, d_model_default


def _download_state_dict(repo_id: str) -> Optional[dict]:
    """Try each known weight filename; return the loaded state dict or None.

    HyenaDNA .ckpt files embed an OmegaConf object, so weights_only=False is
    required (these are the authors' official public checkpoints).
    """
    from huggingface_hub import hf_hub_download

    for fname in _WEIGHT_FILES:
        try:
            path = hf_hub_download(repo_id, fname)
        except Exception:  # noqa: BLE001 — try the next candidate filename
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return None
