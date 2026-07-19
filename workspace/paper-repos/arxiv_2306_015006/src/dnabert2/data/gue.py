"""GUE benchmark loader — configurable across all 28 GUE datasets.

The Genome Understanding Evaluation (GUE) benchmark is introduced by DNABERT-2
(Sec 4.2 / Table 12). Each (task, subset) provides train/dev/test CSVs with
columns [sequence, label]. This loader reads local CSVs populated by
data/download.py, falling back to the HF `datasets` hub.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class GUEDataset(Dataset):
    """A (sequence, label) dataset for one GUE (task, subset) split.

    Args:
        sequences: raw DNA strings.
        labels: integer class labels.
    """

    def __init__(self, sequences: List[str], labels: List[int]) -> None:
        assert len(sequences) == len(labels), "sequences/labels length mismatch"
        self.sequences = sequences
        self.labels = labels

    def __repr__(self) -> str:  # noqa: D105
        return f"GUEDataset(n={len(self)})"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.sequences[idx], int(self.labels[idx])


def _read_csv(path: str) -> Tuple[List[str], List[int]]:
    import pandas as pd

    df = pd.read_csv(path)
    # GUE CSVs use columns 'sequence' and 'label'.
    seq_col = "sequence" if "sequence" in df.columns else df.columns[0]
    lab_col = "label" if "label" in df.columns else df.columns[-1]
    return df[seq_col].astype(str).tolist(), df[lab_col].astype(int).tolist()


def load_gue_split(task: str, subset: str, split: str, data_dir: str) -> Tuple[List[str], List[int]]:
    """Load one GUE split from local CSV, or fall back to HF datasets.

    Local layout (written by data/download.py):
        {data_dir}/GUE/{task}/{subset}/{split}.csv
    """
    local = os.path.join(data_dir, "GUE", task, subset, f"{split}.csv")
    if os.path.exists(local):
        return _read_csv(local)

    # Fallback: HF datasets hosting the GUE benchmark. Config names on the hub
    # differ from the paper's task names (e.g. promoter_detection/all ->
    # prom_300_all), so map through the registry.
    try:
        from datasets import load_dataset

        from ..utils.config import hf_config_name

        cfg = hf_config_name(task, subset)
        hf_split = {"train": "train", "dev": "validation", "test": "test"}.get(split, split)
        try:
            ds = load_dataset("leannmlindsey/GUE", name=cfg, split=hf_split)
        except ValueError:
            # Some configs expose 'dev' rather than 'validation'.
            alt = {"validation": "dev", "dev": "validation"}.get(hf_split, hf_split)
            ds = load_dataset("leannmlindsey/GUE", name=cfg, split=alt)
        seq_col = "sequence" if "sequence" in ds.column_names else ds.column_names[0]
        return list(ds[seq_col]), list(ds["label"])
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"No local CSV at {local} and HF fallback failed: {exc}\n"
            f"Run: python data/download.py --task {task}"
        ) from exc


def collate_factory(tokenizer, max_len: int):
    """Build a collate_fn that BPE-tokenizes a batch of (seq, label)."""
    def collate(batch):
        seqs, labels = zip(*batch)
        enc = tokenizer.encode_batch(list(seqs), max_len=max_len)
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate
