"""Character-level DNA tokenizer for Caduceus (Sec 5.1).

Caduceus uses character-/base-pair-level tokenization (A/C/G/T/N + special
tokens), avoiding the k-mer instability discussed in the paper. When the
official tokenizer ships with the HF weights we defer to it; otherwise this
minimal char tokenizer reproduces the same vocab contract.
"""
from __future__ import annotations

from typing import Dict, List

import torch

# Vocab layout mirrors the Caduceus char tokenizer: specials first, then bases.
_SPECIAL = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
_BASES = ["A", "C", "G", "T", "N"]
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def reverse_complement(seq: str) -> str:
    """Reverse complement of a DNA string (A<->T, C<->G), reversed 5'->3'."""
    return "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq.upper()))


class CharTokenizer:
    """Minimal char-level DNA tokenizer with an RC helper.

    If ``hf_name`` is given and its tokenizer loads, that (official) tokenizer
    is used and its pad id honoured; otherwise the built-in char vocab is used.
    """

    def __init__(self, hf_name: str | None = None, max_len: int = 1024) -> None:
        self.max_len = max_len
        self.hf = None
        if hf_name is not None:
            try:
                from transformers import AutoTokenizer

                self.hf = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            except Exception:
                self.hf = None  # fall back to the built-in char vocab

        vocab = _SPECIAL + _BASES
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.pad_id = self.stoi["[PAD]"]
        self.unk_id = self.stoi["[UNK]"]

    @property
    def vocab_size(self) -> int:
        if self.hf is not None:
            return int(self.hf.vocab_size)
        return len(self.stoi)

    def complement_id_map(self) -> torch.Tensor:
        """Map each vocab id to its complement id (specials map to themselves).

        Used to build the reverse complement of a *tokenised* sequence for
        Caduceus-Ph post-hoc conjoining.
        """
        n = self.vocab_size
        comp = torch.arange(n)
        if self.hf is None:
            for b, c in _COMPLEMENT.items():
                if b in self.stoi and c in self.stoi:
                    comp[self.stoi[b]] = self.stoi[c]
        else:
            for b, c in _COMPLEMENT.items():
                ib = self.hf.convert_tokens_to_ids(b)
                ic = self.hf.convert_tokens_to_ids(c)
                if ib is not None and ic is not None and ib >= 0 and ic >= 0:
                    comp[ib] = ic
        return comp

    def encode_batch(self, seqs: List[str], max_len: int | None = None) -> Dict[str, torch.Tensor]:
        max_len = max_len or self.max_len
        if self.hf is not None:
            enc = self.hf(
                list(seqs), padding="max_length", truncation=True,
                max_length=max_len, return_tensors="pt",
            )
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc.get("attention_mask",
                                              (enc["input_ids"] != self.pad_id).long())}

        ids, masks = [], []
        for s in seqs:
            toks = [self.stoi.get(ch, self.unk_id) for ch in s.upper()[:max_len]]
            attn = [1] * len(toks)
            pad = max_len - len(toks)
            toks += [self.pad_id] * pad
            attn += [0] * pad
            ids.append(toks)
            masks.append(attn)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long)}
