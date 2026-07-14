"""Character-level single-nucleotide tokenizer.

Implements the tokenization described in HyenaDNA (Sec 3): each nucleotide maps
to one token at single-nucleotide resolution. SIR confidence 0.85; the exact
special-token ids follow the official repo convention (pad id 4).

NOTE: When loading official pretrained weights via HuggingFace, prefer the
tokenizer bundled with the model (see models/pretrained.py). This class is a
dependency-free fallback used for the from-scratch path and for inference when
the HF tokenizer is unavailable.
"""
from __future__ import annotations

from typing import Dict, List

# Vocabulary MUST match the official HyenaDNA CharacterTokenizer exactly, or the
# pretrained embeddings receive wrong token ids. Special tokens 0-6, then the
# nucleotides start at id 7 (characters=['A','C','G','T','N']). [PAD]=4.
_VOCAB: Dict[str, int] = {
    "[CLS]": 0,
    "[SEP]": 1,
    "[BOS]": 2,
    "[MASK]": 3,
    "[PAD]": 4,
    "[RESERVED]": 5,
    "[UNK]": 6,
    "A": 7,
    "C": 8,
    "G": 9,
    "T": 10,
    "N": 11,
}


class CharTokenizer:
    """Maps a DNA string to a list of integer token ids.

    Paper reference: HyenaDNA Sec 3 (single-nucleotide, character-level).

    Args:
        max_len: sequences longer than this are truncated; shorter are padded.
        add_special: whether to wrap with [BOS]/[SEP] (kept off by default to
            match simple classification pooling).
    """

    def __init__(self, max_len: int = 1024, add_special: bool = False) -> None:
        self.max_len = max_len
        self.add_special = add_special
        self.vocab = dict(_VOCAB)
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]

    def __repr__(self) -> str:  # noqa: D105
        return f"CharTokenizer(vocab_size={len(self.vocab)}, max_len={self.max_len})"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, seq: str, max_len: int | None = None) -> List[int]:
        """Encode a raw DNA string to padded/truncated token ids."""
        max_len = max_len or self.max_len
        seq = seq.upper().strip()
        ids: List[int] = []
        if self.add_special:
            ids.append(self.vocab["[BOS]"])
        for ch in seq:
            ids.append(self.vocab.get(ch, self.unk_id))
        if self.add_special:
            ids.append(self.vocab["[SEP]"])
        # Truncate then pad to fixed length.
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids
