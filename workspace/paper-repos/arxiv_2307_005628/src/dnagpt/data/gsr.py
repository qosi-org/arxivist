"""GSR (genomic signal & region) dataset loader (Sec 4, S1.4.1).

Following DeepGSR: PAS (polyadenylation signal) and TIS (translation initiation
site) recognition for human/mouse/bovine/fruit fly. Each example is 300 bp
before + 300 bp after the GSR (the motif itself is removed in preprocessing, so
the model must use flanking non-coding context). Binary classification.

Reads locally-prepared CSVs first; if absent, falls back to a small SYNTHETIC
generator so the pipeline + tests run without the full multi-genome download.
The synthetic task embeds a weak GC-skew signal in positives — enough to verify
the model learns, NOT to reproduce the paper's accuracy.
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

_BASES = "ACGT"


class GSRDataset(Dataset):
    """A (sequence, label) dataset for one GSR split."""

    def __init__(self, sequences: List[str], labels: List[int]) -> None:
        assert len(sequences) == len(labels), "sequences/labels length mismatch"
        self.sequences = sequences
        self.labels = labels

    def __repr__(self) -> str:  # noqa: D105
        return f"GSRDataset(n={len(self)})"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.sequences[idx], int(self.labels[idx])


#: True whenever the current split came from the synthetic fallback (not real data).
USING_SYNTHETIC = False


def _synthetic_gsr(n: int, length: int, seed: int) -> Tuple[List[str], List[int]]:
    """Toy PAS-like task with a DELIBERATELY WEAK, noisy signal.

    This is a smoke-test generator, NOT a stand-in for DeepGSR. The signal is a
    tiny GC-skew plus 15% label noise, so a competent model lands well below 1.0
    and a perfect score is impossible — making it obvious this is not the paper's
    task. Do not read any accuracy here as a reproduction of Table S2.
    """
    rng = random.Random(seed)
    seqs, labels = [], []
    for i in range(n):
        pos = i % 2 == 0
        # very mild GC vs AT skew (not separable to 100%)
        weights = [24, 26, 26, 24] if pos else [26, 24, 24, 26]
        s = "".join(rng.choices(_BASES, weights=weights, k=length))
        label = 1 if pos else 0
        if rng.random() < 0.15:          # 15% label noise -> caps achievable accuracy
            label = 1 - label
        seqs.append(s)
        labels.append(label)
    return seqs, labels


def load_gsr_split(task: str, split: str, data_dir: str,
                   length: int = 606, synthetic_n: int = 512) -> Tuple[List[str], List[int]]:
    """Load a GSR split from CSV, else synthesize.

    CSV layout: ``data_dir/gsr/<task>/<split>.csv`` with columns ``sequence,label``.
    """
    global USING_SYNTHETIC
    csv_path = os.path.join(data_dir, "gsr", task, f"{split}.csv")
    if os.path.isfile(csv_path):
        import pandas as pd

        df = pd.read_csv(csv_path)
        return df["sequence"].astype(str).tolist(), df["label"].astype(int).tolist()

    # synthetic fallback (deterministic per split)
    USING_SYNTHETIC = True
    seed = {"train": 0, "val": 1, "test": 2}.get(split, 3) + hash(task) % 1000
    n = synthetic_n if split == "train" else synthetic_n // 4
    print("=" * 74)
    print(f"[gsr] !! NO REAL DATA at {csv_path} -> SYNTHETIC {split} (n={n}).")
    print("[gsr] !! This is a SMOKE TEST only. Its accuracy is NOT comparable to")
    print("[gsr] !! the paper (Table S2, 91.51%). Get real DeepGSR PAS/TIS data —")
    print("[gsr] !! see data/README_data.md — to reproduce the paper's numbers.")
    print("=" * 74)
    return _synthetic_gsr(n, length, seed)


def collate_factory(tokenizer, species: str, max_len: int):
    """Build a collate_fn that tokenizes a batch of (seq, label) via the token language."""
    def collate(batch):
        seqs, labels = zip(*batch)
        enc = tokenizer.encode_batch(list(seqs), species=species, max_len=max_len)
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate
