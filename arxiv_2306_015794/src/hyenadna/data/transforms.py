"""Sequence transforms for genomic data.

Reverse-complement augmentation is mentioned as an optional strategy in the SIR
(training_pipeline.data_augmentation, conf 0.62). Applied on raw strings before
tokenization.
"""
from __future__ import annotations

_COMPLEMENT = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"})


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string."""
    return seq.upper().translate(_COMPLEMENT)[::-1]
