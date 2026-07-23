"""Dataset loaders for the genomic benchmarks GENERator is evaluated on.

Default: Genomic Benchmarks (paper Table S5). Also supports Nucleotide
Transformer tasks via HF datasets. Reads local CSVs first, then falls back to
each benchmark's API.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class GenomicDataset(Dataset):
    """A (sequence, label) dataset for one benchmark split."""

    def __init__(self, sequences: List[str], labels: List[int]) -> None:
        assert len(sequences) == len(labels), "sequences/labels length mismatch"
        self.sequences = sequences
        self.labels = labels

    def __repr__(self) -> str:  # noqa: D105
        return f"GenomicDataset(n={len(self)})"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.sequences[idx], int(self.labels[idx])


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


def load_nt_task(task: str, split: str, data_dir: str) -> Tuple[List[str], List[int], int]:
    """Load a Nucleotide Transformer benchmark task via HF datasets."""
    try:
        from datasets import load_dataset

        hf_split = {"train": "train", "test": "test"}.get(split, split)
        ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", task, split=hf_split)
        return list(ds["sequence"]), list(ds["label"]), len(set(ds["label"]))
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not load NT task '{task}': {exc}\n"
            f"Run: python data/download.py --benchmark nt_tasks --task {task}"
        ) from exc


def load_split(benchmark: str, task: str, split: str, data_dir: str) -> Tuple[List[str], List[int], int]:
    """Dispatch to the right benchmark loader."""
    if benchmark == "genomic_benchmarks":
        return load_genomic_benchmarks(task, split, data_dir)
    if benchmark == "nt_tasks":
        return load_nt_task(task, split, data_dir)
    raise ValueError(f"Unknown benchmark {benchmark!r}")


def collate_factory(tokenizer, max_len: int):
    """Build a collate_fn that 6-mer-tokenizes a batch of (seq, label)."""
    def collate(batch):
        seqs, labels = zip(*batch)
        enc = tokenizer.encode_batch(list(seqs), max_len=max_len)
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate
