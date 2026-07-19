"""Wrapper over the official DNABERT-2 BPE tokenizer.

DNABERT-2 replaces k-mer tokenization with SentencePiece Byte Pair Encoding
(paper Sec 3.1, vocab 4096). We load the tokenizer bundled with the pretrained
model rather than reimplementing BPE, so token ids match the released weights.
"""
from __future__ import annotations

from typing import Dict, List

import torch


class DNATokenizer:
    """Thin wrapper around the HuggingFace DNABERT-2 tokenizer.

    Args:
        model_name: HF repo id of the DNABERT-2 model/tokenizer.
        max_len: max sequence length (BPE tokens) for padding/truncation.
    """

    def __init__(self, model_name: str = "zhihan1996/DNABERT-2-117M", max_len: int = 128) -> None:
        from transformers import AutoTokenizer

        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def __repr__(self) -> str:  # noqa: D105
        return f"DNATokenizer(vocab_size={self.tok.vocab_size}, max_len={self.max_len})"

    def encode_batch(self, seqs: List[str], max_len: int | None = None) -> Dict[str, torch.Tensor]:
        """BPE-tokenize a batch of DNA strings to input_ids + attention_mask."""
        out = self.tok(
            list(seqs),
            padding="max_length",
            truncation=True,
            max_length=max_len or self.max_len,
            return_tensors="pt",
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}
