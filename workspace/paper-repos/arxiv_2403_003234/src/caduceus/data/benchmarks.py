"""Dataset loaders for the Genomic Benchmarks Caduceus is evaluated on (Sec 5.2.1).

Default: Genomic Benchmarks (Gresova et al. 2023, 8 regulatory-element tasks).
Reads locally-downloaded data first, then falls back to the genomic_benchmarks
API. Supports RC data augmentation for Caduceus-Ph training.
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import reverse_complement


class GenomicDataset(Dataset):
    """A (sequence, label) dataset for one benchmark split.

    Args:
        sequences, labels: aligned lists.
        rc_aug: if True, each __getitem__ returns the reverse complement with
            probability 0.5 (RC data augmentation, Sec 5.1 / used by Caduceus-Ph).
    """

    def __init__(self, sequences: List[str], labels: List[int], rc_aug: bool = False) -> None:
        assert len(sequences) == len(labels), "sequences/labels length mismatch"
        self.sequences = sequences
        self.labels = labels
        self.rc_aug = rc_aug

    def __repr__(self) -> str:  # noqa: D105
        return f"GenomicDataset(n={len(self)}, rc_aug={self.rc_aug})"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        seq = self.sequences[idx]
        if self.rc_aug and random.random() < 0.5:
            seq = reverse_complement(seq)
        return seq, int(self.labels[idx])


def load_genomic_benchmarks(task: str, split: str, data_dir: str) -> Tuple[List[str], List[int], int]:
    """Load a Genomic Benchmarks dataset via the genomic_benchmarks API."""
    base = os.path.join(data_dir, "genomic_benchmarks", task)
    split_dir = os.path.join(base, "train" if split == "train" else "test")
    if not os.path.isdir(split_dir):
        try:
            from genomic_benchmarks.loc2seq import download_dataset

            download_dataset(task, dest_path=os.path.join(data_dir, "genomic_benchmarks"))
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                f"Could not load Genomic Benchmarks task '{task}': {exc}\n"
                f"Run: python data/download.py --benchmark genomic_benchmarks --task {task}"
            ) from exc
    seqs, labels, classes = [], [], sorted(os.listdir(split_dir))
    for ci, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        for fn in os.listdir(cls_dir):
            with open(os.path.join(cls_dir, fn), "r", encoding="utf-8") as fh:
                seqs.append(fh.read().strip())
                labels.append(ci)
    return seqs, labels, len(classes)


def load_split(benchmark: str, task: str, split: str, data_dir: str) -> Tuple[List[str], List[int], int]:
    """Dispatch to the right benchmark loader."""
    if benchmark == "genomic_benchmarks":
        return load_genomic_benchmarks(task, split, data_dir)
    raise ValueError(f"Unknown benchmark {benchmark!r}")


def collate_factory(tokenizer, max_len: int):
    """Build a collate_fn that char-tokenizes a batch of (seq, label)."""
    def collate(batch):
        seqs, labels = zip(*batch)
        enc = tokenizer.encode_batch(list(seqs), max_len=max_len)
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate
