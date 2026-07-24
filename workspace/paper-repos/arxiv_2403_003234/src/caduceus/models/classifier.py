"""Caduceus backbone + classification head.

Reproduction critical path: load the official `kuleshov-group/caduceus-*`
weights from the HuggingFace Hub via `AutoModel(trust_remote_code=True)`. The
backbone is a bi-directional, RC-equivariant Mamba LM; its real forward pass
uses the fused `mamba-ssm` + `causal-conv1d` CUDA kernels (GPU required).

For classification we **mean-pool** the final hidden states and apply a linear
head (Sec 5.2.1 / Appendix D.1). Two equivariance modes:

* ``variant="ph"`` (Caduceus-Ph): post-hoc conjoining — at inference run the
  model on the sequence and its reverse complement and average (RC-ensemble).
* ``variant="ps"`` (Caduceus-PS): parameter sharing — the final hidden dim is
  channel-split; average the two halves to enforce RC-invariance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CaduceusClassifier(nn.Module):
    """Caduceus backbone with a mean-pool + linear classification head.

    Args:
        backbone: pretrained Caduceus model returning hidden states [B, L, D].
        d_model: backbone hidden size (read from its config, never hardcoded).
        num_classes: number of output classes.
        variant: "ph" (post-hoc conjoining) or "ps" (parameter sharing).
        rc_complement_ids: (input_id -> complement_id) map for building the RC
            of a token sequence, needed for Caduceus-Ph conjoining.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        num_classes: int,
        variant: str = "ph",
        rc_complement_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.variant = variant.lower()
        # Caduceus-PS pools each channel-split half then averages, so the head
        # sees D/2 features; Caduceus-Ph pools the full D.
        head_in = d_model // 2 if self.variant == "ps" else d_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(head_in, num_classes)
        if rc_complement_ids is not None:
            self.register_buffer("rc_complement_ids", rc_complement_ids)
        else:
            self.rc_complement_ids = None

    def __repr__(self) -> str:  # noqa: D105
        return f"CaduceusClassifier(variant={self.variant}, head={self.classifier})"

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int,
        variant: str = "ph",
        device: str = "cuda",
        rc_complement_ids: torch.Tensor | None = None,
    ) -> "CaduceusClassifier":
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        d_model = getattr(config, "d_model", None) or getattr(config, "hidden_size", 256)

        backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, output_hidden_states=True
        )
        model = cls(
            backbone, d_model=d_model, num_classes=num_classes,
            variant=variant, rc_complement_ids=rc_complement_ids,
        )
        return model.to(device)

    # --- helpers -----------------------------------------------------------
    def _hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return out.hidden_states[-1]
        return out[0]

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Masked mean pool over the length dim -> [B, D]."""
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        if self.variant == "ps":
            # RC-invariance: split channels and average the two halves.
            half = pooled.size(-1) // 2
            pooled = 0.5 * (pooled[..., :half] + pooled[..., half:])
        return pooled

    def _logits_single(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        hidden = self._hidden(input_ids)
        pooled = self._pool(hidden, attention_mask)
        pooled = pooled.to(self.classifier.weight.dtype)
        return self.classifier(self.dropout(pooled))

    def _rc_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.rc_complement_ids is not None, "rc_complement_ids required for conjoining"
        return self.rc_complement_ids[torch.flip(input_ids, dims=(1,))]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        conjoin: bool = False,
    ) -> torch.Tensor:
        assert input_ids.dim() == 2, f"Expected [B, L], got {tuple(input_ids.shape)}"
        logits = self._logits_single(input_ids, attention_mask)
        # Caduceus-Ph post-hoc conjoining: average with the RC pass (inference).
        if conjoin and self.variant == "ph" and self.rc_complement_ids is not None:
            rc_ids = self._rc_input_ids(input_ids)
            rc_mask = torch.flip(attention_mask, dims=(1,)) if attention_mask is not None else None
            logits = 0.5 * (logits + self._logits_single(rc_ids, rc_mask))
        return logits
