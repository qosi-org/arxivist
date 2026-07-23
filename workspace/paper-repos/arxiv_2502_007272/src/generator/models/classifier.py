"""GENERator backbone + classification head.

Reproduction critical path: load the official
`GenerTeam/GENERator-eukaryote-1.2b-base` (a LLaMA decoder-only causal LM,
model_type=llama, confirmed 26L/2048d/32H-4KV/vocab4128). It loads cleanly via
AutoModel (no custom kernels). For classification we pool the **last non-pad
token (<EOS>) embedding** and apply a linear head, per the paper's Appendix C.4:
"For causal language models, predictions derived from the <EOS> token embedding
via a linear layer."
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GENERatorClassifier(nn.Module):
    """GENERator causal-LM backbone with a linear classification head.

    Args:
        backbone: the pretrained GENERator model (hidden states [B, L, D]).
        d_model: backbone hidden size.
        num_classes: number of output classes.
    """

    def __init__(self, backbone: nn.Module, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)

    def __repr__(self) -> str:  # noqa: D105
        return f"GENERatorClassifier(head={self.classifier})"

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int,
        device: str = "cuda",
        load_in_8bit: bool = False,
    ) -> "GENERatorClassifier":
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        d_model = getattr(config, "hidden_size", 2048)

        kwargs = {"trust_remote_code": True, "output_hidden_states": True}
        if load_in_8bit:
            # Optional: fit the 1.2B model on tighter GPUs (needs bitsandbytes).
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        backbone = AutoModel.from_pretrained(model_name, **kwargs)
        model = cls(backbone, d_model=d_model, num_classes=num_classes)
        # With device_map='auto' (8-bit) the backbone is already placed.
        return model if load_in_8bit else model.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert input_ids.dim() == 2, f"Expected [B, L], got {tuple(input_ids.shape)}"
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        # Last non-pad token per sequence (the <EOS>/final content token).
        # attention_mask sums to the true length; index = length - 1.
        lengths = attention_mask.long().sum(dim=1) - 1  # [B]
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))  # [B,1,D]
        pooled = hidden.gather(1, idx).squeeze(1)  # [B, D]

        pooled = pooled.to(self.classifier.weight.dtype)
        return self.classifier(self.dropout(pooled))
