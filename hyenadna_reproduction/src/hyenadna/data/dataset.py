"""Torch Dataset wrappers for genomic classification benchmarks.

Supports GenomicBenchmarks (via the `genomic_benchmarks` API) and Nucleotide
Transformer tasks (via HuggingFace `datasets`). All paths flow from config —
nothing hardcoded.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


class GenomicDataset(Dataset):
    """A tokenized (sequence, label) dataset.

    Args:
        sequences: raw DNA strings.
        labels: integer class labels.
        tokenizer: CharTokenizer instance.
        max_len: sequence length after tokenization.
    """

    def __init__(self, sequences: List[str], labels: List[int], tokenizer: CharTokenizer, max_len: int) -> None:
        assert len(sequences) == len(labels), "sequences and labels length mismatch"
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __repr__(self) -> str:  # noqa: D105
        return f"GenomicDataset(n={len(self)}, max_len={self.max_len})"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ids = self.tokenizer.encode(self.sequences[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), int(self.labels[idx])


def load_genomic_benchmarks(dataset: str, split: str, data_dir: str) -> Tuple[List[str], List[int], int]:
    """Load a GenomicBenchmarks dataset into (sequences, labels, num_classes).

    Reads from the local data_dir populated by data/download.py. Falls back to
    the genomic_benchmarks API if the local cache is missing.
    """
    local = os.path.join(data_dir, "genomic_benchmarks", dataset, f"{split}.tsv")
    if os.path.exists(local):
        seqs, labels = [], []
        with open(local, "r", encoding="utf-8") as f:
            next(f, None)  # header
            for line in f:
                seq, lab = line.rstrip("\n").split("\t")
                seqs.append(seq)
                labels.append(int(lab))
        return seqs, labels, len(set(labels))

    # API fallback
    try:
        from genomic_benchmarks.loc2seq import download_dataset
        from genomic_benchmarks.data_check import list_datasets  # noqa: F401

        base = download_dataset(dataset, dest_path=os.path.join(data_dir, "genomic_benchmarks"))
        split_dir = os.path.join(str(base), split)
        seqs, labels, classes = [], [], sorted(os.listdir(split_dir))
        for ci, cls in enumerate(classes):
            cls_dir = os.path.join(split_dir, cls)
            for fn in os.listdir(cls_dir):
                with open(os.path.join(cls_dir, fn), "r", encoding="utf-8") as fh:
                    seqs.append(fh.read().strip())
                    labels.append(ci)
        return seqs, labels, len(classes)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"No local cache at {local} and genomic_benchmarks API failed: {exc}\n"
            f"Run: python data/download.py genomic-benchmarks --dataset {dataset}"
        ) from exc


def load_nt_benchmark(dataset: str, split: str, data_dir: str) -> Tuple[List[str], List[int], int]:
    """Load a Nucleotide Transformer benchmark task via HuggingFace datasets."""
    try:
        from datasets import load_dataset

        ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", dataset, split=split)
        seqs = list(ds["sequence"])
        labels = list(ds["label"])
        return seqs, labels, len(set(labels))
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not load NT task '{dataset}' split '{split}': {exc}\n"
            f"Run: python data/download.py nt-benchmarks --dataset {dataset}"
        ) from exc


def build_dataset(cfg_data: dict, split: str, tokenizer: CharTokenizer) -> Tuple[GenomicDataset, int]:
    """Dispatch to the right loader based on config data.source."""
    source = cfg_data["source"]
    name = cfg_data["dataset"]
    data_dir = cfg_data.get("data_dir", "data/")
    max_len = cfg_data.get("max_len", 1024)

    if source == "genomic-benchmarks":
        seqs, labels, n_cls = load_genomic_benchmarks(name, split, data_dir)
    elif source == "nt-benchmarks":
        seqs, labels, n_cls = load_nt_benchmark(name, split, data_dir)
    else:
        raise ValueError(f"Unknown data.source={source!r}")
    return GenomicDataset(seqs, labels, tokenizer, max_len), n_cls
