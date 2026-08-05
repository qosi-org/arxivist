"""
models/baselines/chronos2_wrapper.py
=======================================
Chronos-2 zero-shot baseline (Section 5.5): pretrained foundation model used
as a frozen feature extractor for a univariate target series, with a
trainable linear head mapping embeddings to the final prediction. Loaded via
Hugging Face transformers, per the paper's explicit statement.

NOTE: Chronos-2 checkpoint availability/exact model name is not given in the
paper beyond "Chronos-2"; this wrapper assumes the community-standard
Amazon Chronos model family naming convention and requires network access
to Hugging Face Hub to download weights (see README "Known Limitations").
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Chronos2Baseline(nn.Module):
    """
    Args:
        embedding_dim: dimensionality of the frozen Chronos-2 embedding used
            by the trainable linear head. Model-dependent; exposed as a
            config parameter since the paper does not specify a checkpoint
            name/size.
        model_name: Hugging Face Hub identifier for the Chronos-2 checkpoint.
            ASSUMED default; paper does not give an exact identifier.
    """

    def __init__(self, embedding_dim: int = 512, model_name: str = "amazon/chronos-t5-small") -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._backbone = None  # lazily loaded (large download; avoid import-time cost)
        self.head = nn.Linear(embedding_dim, 1)

    def _load_backbone(self):
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Chronos2Baseline requires 'transformers'. Install with: pip install transformers"
            ) from exc
        self._backbone = AutoModel.from_pretrained(self.model_name)
        for p in self._backbone.parameters():
            p.requires_grad_(False)  # frozen feature extractor, per paper

    def forward(self, univariate_series: torch.Tensor) -> torch.Tensor:
        """
        Args:
            univariate_series: [B, L] univariate target-only series (the
                paper extracts only the target values for Chronos-2 input).
        Returns:
            [B] scalar predictions via frozen embedding + trainable linear head.
        """
        if self._backbone is None:
            self._load_backbone()
        with torch.no_grad():
            # Placeholder embedding extraction: real Chronos-2 usage requires
            # its specific tokenizer/quantization scheme (values are
            # quantized into a finite vocabulary per Section 5.5). This
            # wrapper exposes the integration point; consult the actual
            # Chronos-2 model card for the exact encode() API.
            embedding = self._backbone.encoder(univariate_series.unsqueeze(-1)).last_hidden_state.mean(dim=1)
        return self.head(embedding).squeeze(-1)
