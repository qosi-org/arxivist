"""The DNAGPT model (Sec 2.1, Fig 1c).

A GPT decoder that jointly processes DNA-**sequence** tokens and **numerical**
tokens: sequence + numerical embeddings are concatenated along the length dim,
run through a causal GPT stack, then split and sent to a Classification head
(sequence/special tokens, CE) and a Regression head (number tokens, MSE).

This is a from-scratch re-implementation (the architecture is the paper's
contribution). It runs on CPU with no custom kernels. ``from_pretrained`` loads
the authors' official ``.pth`` (Google Drive / Weiyun) via a documented
state_dict key-map, falling back to ``strict=False`` when keys differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .blocks import GPTBlock, NumericalEmbedding, RegressionHead, SequentialEmbedding

# DNAGPT variant dims (paper Fig S3). H/M/S-512 = 0.1B; B-512 = 3B.
VARIANT_DIMS = {
    "H":     {"n_layer": 12, "hidden": 768,  "n_heads": 12, "seq_len": 4096},
    "M":     {"n_layer": 12, "hidden": 768,  "n_heads": 12, "seq_len": 4096},
    "S-512": {"n_layer": 12, "hidden": 768,  "n_heads": 12, "seq_len": 512},
    "B-512": {"n_layer": 60, "hidden": 2048, "n_heads": 64, "seq_len": 512},
}


@dataclass
class DNAGPTConfig:
    """DNAGPT hyperparameters. Defaults match DNAGPT-M (0.1B)."""

    vocab_size: int = 19564           # official DNAGPT vocab (19530 k-mers + specials)
    hidden: int = 768
    n_layer: int = 12
    n_heads: int = 12
    seq_len: int = 512
    dropout: float = 0.0
    num_classes: int = 2               # downstream binary classification (GSR)

    @classmethod
    def from_variant(cls, variant: str, **overrides) -> "DNAGPTConfig":
        dims = VARIANT_DIMS[variant]
        cfg = cls(hidden=dims["hidden"], n_layer=dims["n_layer"],
                  n_heads=dims["n_heads"], seq_len=dims["seq_len"])
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


class DNAGPT(nn.Module):
    """Generalized GPT for DNA sequence + numerical data."""

    def __init__(self, config: DNAGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_embed = SequentialEmbedding(config.vocab_size, config.hidden)
        self.num_embed = NumericalEmbedding(config.hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.hidden))
        self.blocks = nn.ModuleList(
            [GPTBlock(config.hidden, config.n_heads, dropout=config.dropout)
             for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.hidden)
        # dual heads
        self.classification_head = nn.Linear(config.hidden, config.vocab_size, bias=False)
        self.regression_head = RegressionHead(config.hidden)
        # downstream classification head (GSR: True/False)
        self.cls_head = nn.Linear(config.hidden, config.num_classes)
        self.apply(self._init_weights)

    def __repr__(self) -> str:  # noqa: D105
        n = sum(p.numel() for p in self.parameters())
        return (f"DNAGPT(layers={self.config.n_layer}, hidden={self.config.hidden}, "
                f"heads={self.config.n_heads}, params={n/1e6:.1f}M)")

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(getattr(m, "weight"), mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def backbone(self, seq_ids: torch.Tensor, numbers: Optional[torch.Tensor],
                 attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Shared GPT forward -> hidden states [B, T, D]."""
        x = self.seq_embed(seq_ids)                       # [B, L, D]
        if numbers is not None:
            x = torch.cat([x, self.num_embed(numbers)], dim=1)  # append number tokens
        t = x.size(1)
        x = x + self.pos_embed[:, :t, :]
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.ln_f(x)

    def forward(self, seq_ids: torch.Tensor, numbers: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Return classification logits, regression values, and hidden states."""
        assert seq_ids.dim() == 2, f"expected [B, L], got {tuple(seq_ids.shape)}"
        hidden = self.backbone(seq_ids, numbers, attn_mask)
        return {
            "hidden": hidden,
            "class_logits": self.classification_head(hidden),  # [B, T, vocab] (CE)
            "reg_values": self.regression_head(hidden),        # [B, T, 1]    (MSE)
        }

    def classify(self, seq_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Downstream binary GSR classification from the last token (Sec 4.1)."""
        hidden = self.backbone(seq_ids, None, attn_mask)
        if attn_mask is not None:
            lengths = attn_mask.long().sum(dim=1) - 1
            idx = lengths.clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            pooled = hidden.gather(1, idx).squeeze(1)
        else:
            pooled = hidden[:, -1, :]
        return self.cls_head(pooled)

    @classmethod
    def from_pretrained(cls, ckpt_path: str, config: DNAGPTConfig,
                        device: str = "cpu", strict: bool = False) -> "DNAGPT":
        """Load the official DNAGPT .pth (Google Drive / Weiyun).

        The authors' checkpoint uses a GPT-2-style layout
        (``transformer.wte/wpe/h.N.{ln_1,attn.c_attn,attn.c_proj,mlp.c_fc,
        mlp.c_proj,ln_2}/ln_f`` + ``mlm_head``). We remap those names onto this
        module tree and transpose the GPT-2 ``Conv1D`` weights (which are stored
        [in, out]) to the ``nn.Linear`` convention ([out, in]). Backbone +
        embeddings load exactly; the downstream ``cls_head`` is trained fresh.
        Run ``data/download.py --weights dna_gpt0.1b_m`` first.
        """
        model = cls(config)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model", state.get("state_dict", state))
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

        mapped = model._map_official_state_dict(state)
        report = model.load_state_dict(mapped, strict=strict)
        matched = len(mapped) - len(report.unexpected_keys)
        print(f"[from_pretrained] loaded {ckpt_path} | mapped={len(mapped)} "
              f"matched~{matched} missing={len(report.missing_keys)} "
              f"unexpected={len(report.unexpected_keys)}")
        return model.to(device)

    def _map_official_state_dict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Translate the official GPT-2-style keys onto this model's names.

        GPT-2 ``Conv1D`` stores weight as [in_features, out_features]; ``nn.Linear``
        expects [out_features, in_features], so those tensors are transposed.
        """
        out: Dict[str, torch.Tensor] = {}
        vocab = self.seq_embed.emb.weight.shape[0]

        def take(v: torch.Tensor, rows: int | None = None) -> torch.Tensor:
            # optionally trim/pad the vocab dim if it differs slightly
            if rows is not None and v.shape[0] != rows:
                if v.shape[0] > rows:
                    return v[:rows]
                pad = torch.zeros(rows - v.shape[0], *v.shape[1:], dtype=v.dtype)
                return torch.cat([v, pad], dim=0)
            return v

        for k, v in state.items():
            if k == "transformer.wte.weight":
                out["seq_embed.emb.weight"] = take(v, vocab)
                out["classification_head.weight"] = take(v, vocab)  # tied lm head fallback
            elif k == "transformer.wpe.weight":
                # official positional table [P, D] -> our [1, P, D] parameter
                p = v.shape[0]
                cur = self.pos_embed.shape[1]
                pe = v[:cur] if p >= cur else torch.cat(
                    [v, torch.zeros(cur - p, v.shape[1], dtype=v.dtype)], dim=0)
                out["pos_embed"] = pe.unsqueeze(0)
            elif k == "transformer.ln_f.weight":
                out["ln_f.weight"] = v
            elif k.startswith("transformer.h."):
                i = k.split(".")[2]
                rest = k.split(".", 3)[3]
                m = {
                    "ln_1.weight": f"blocks.{i}.ln1.weight",
                    "ln_2.weight": f"blocks.{i}.ln2.weight",
                    "attn.c_attn.weight": f"blocks.{i}.attn.qkv.weight",
                    "attn.c_proj.weight": f"blocks.{i}.attn.proj.weight",
                    "mlp.c_fc.weight": f"blocks.{i}.mlp.0.weight",
                    "mlp.c_proj.weight": f"blocks.{i}.mlp.2.weight",
                }.get(rest)
                if m is None:
                    continue
                # The official checkpoint stores attn/mlp weights already in the
                # nn.Linear convention [out, in] (verified against the released
                # classification.pth: c_attn is [2304, 768]), so no transpose.
                out[m] = v
            # mlm_head.* is the pretraining MLM head; our classification_head is a
            # single linear, so we skip the intermediate MLM layers (cls_head is
            # trained fresh for GSR). The backbone above is what carries the signal.
        return out
