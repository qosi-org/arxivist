"""DNABERT-2 backbone + classification head.

Reproduction critical path: load the official `zhihan1996/DNABERT-2-117M` via
AutoModel (trust_remote_code=True — it ships ALiBi/GEGLU/Flash-Attn code and has
a proper model_type, so unlike HyenaDNA it loads directly). Attach a linear head
for GUE classification. Paper: Sec 3.2 (architecture), Sec 5 / A.3 (fine-tuning).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DNABERT2Classifier(nn.Module):
    """DNABERT-2 encoder with a linear classification head.

    Args:
        backbone: the pretrained DNABERT-2 model (returns hidden states [B, L, D]).
        d_model: backbone hidden size.
        num_classes: number of output classes.
        pool: 'mean' (masked mean over tokens) or 'cls' (first token).
    """

    def __init__(self, backbone: nn.Module, d_model: int, num_classes: int, pool: str = "mean") -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)

    def __repr__(self) -> str:  # noqa: D105
        return f"DNABERT2Classifier(pool={self.pool}, head={self.classifier})"

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int,
        pool: str = "mean",
        device: str = "cpu",
        attention_dropout: float = 0.1,
    ) -> "DNABERT2Classifier":
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        # DNABERT-2's config.json predates newer transformers and omits
        # `pad_token_id`, but its remote bert_layers.py reads config.pad_token_id
        # in BertEmbeddings -> AttributeError on transformers >= ~4.50, which no
        # longer supplies a default. Recover the true value from the model's own
        # tokenizer ([PAD] = 3) and inject it. Not a guess: read from the repo.
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if getattr(cfg, "pad_token_id", None) is None:
            try:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                pad_id = tok.pad_token_id
            except Exception:  # noqa: BLE001
                pad_id = 3  # DNABERT-2 [PAD] id
            cfg.pad_token_id = pad_id if pad_id is not None else 3

        # Force the PyTorch attention path instead of the Triton Flash-Attn kernel.
        # DNABERT-2's bert_layers.py branches on:
        #     if self.p_dropout or flash_attn_qkvpacked_func is None:
        #         # nonzero attention dropout (e.g. during fine-tuning) -> PyTorch
        # Its shipped flash_attn_triton.py calls tl.dot(..., trans_b=True), an API
        # removed in Triton 3.x (Colab), so the kernel path raises
        # "dot() got an unexpected keyword argument 'trans_b'".
        # A nonzero attention dropout selects the authors' own PyTorch branch —
        # mathematically the same attention, and dropout is what they intend for
        # fine-tuning anyway. attention_dropout is configurable below.
        if float(getattr(cfg, "attention_probs_dropout_prob", 0.0) or 0.0) == 0.0:
            cfg.attention_probs_dropout_prob = attention_dropout

        # `low_cpu_mem_usage=False` is required: newer transformers initialise
        # weights on the `meta` device by default, but DNABERT-2's
        # rebuild_alibi_tensor() eagerly builds real CPU tensors inside
        # __init__, producing "Tensor on device meta is not on the expected
        # device cpu". Disabling lazy init materialises everything on CPU first.
        try:
            backbone = AutoModel.from_pretrained(
                model_name, config=cfg, trust_remote_code=True, low_cpu_mem_usage=False,
            )
        except ImportError as exc:
            if "triton" in str(exc).lower():
                raise ImportError(
                    "DNABERT-2's remote code requires `triton` for Flash-Attention. "
                    "On Colab/Linux-GPU it is preinstalled. Run on a GPU runtime."
                ) from exc
            raise
        except RuntimeError as exc:
            if "meta" in str(exc).lower():
                # Belt-and-braces: some versions need the init context disabled too.
                import torch as _torch

                with _torch.device("cpu"):
                    backbone = AutoModel.from_pretrained(
                        model_name, config=cfg, trust_remote_code=True,
                        low_cpu_mem_usage=False,
                    )
            else:
                raise

        # Hidden size from the model's own config (never hardcoded).
        d_model = getattr(getattr(backbone, "config", cfg), "hidden_size", 768)

        model = cls(backbone, d_model=d_model, num_classes=num_classes, pool=pool)
        return model.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert input_ids.dim() == 2, f"Expected [B, L], got {tuple(input_ids.shape)}"
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # DNABERT-2 returns a tuple; [0] is last_hidden_state [B, L, D].
        hidden = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)

        if self.pool == "mean":
            # Masked mean-pool: ignore padding tokens.
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        elif self.pool == "cls":
            pooled = hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pool={self.pool!r}")

        return self.classifier(self.dropout(pooled))
