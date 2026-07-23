"""Wrapper over the official GENERator 6-mer tokenizer.

GENERator uses non-overlapping 6-mer tokenization (paper Sec 4.2, vocab 4128).
We load the tokenizer bundled with the released weights so token ids match the
pretrained model exactly.
"""
from __future__ import annotations

from typing import Dict, List

import torch


class GenomicTokenizer:
    """Thin wrapper around the HuggingFace GENERator tokenizer.

    Args:
        model_name: HF repo id of the GENERator model/tokenizer.
        max_len: max sequence length in 6-mer tokens.
    """

    def __init__(self, model_name: str = "GenerTeam/GENERator-eukaryote-1.2b-base", max_len: int = 512) -> None:
        from transformers import AutoTokenizer

        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Causal LMs need a pad token for batching; fall back to EOS if unset.
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __repr__(self) -> str:  # noqa: D105
        return f"GenomicTokenizer(vocab_size={self.tok.vocab_size}, max_len={self.max_len})"

    def encode_batch(self, seqs: List[str], max_len: int | None = None) -> Dict[str, torch.Tensor]:
        """6-mer-tokenize a batch of DNA strings to input_ids + attention_mask."""
        out = self.tok(
            list(seqs),
            padding="max_length",
            truncation=True,
            max_length=max_len or self.max_len,
            return_tensors="pt",
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}
