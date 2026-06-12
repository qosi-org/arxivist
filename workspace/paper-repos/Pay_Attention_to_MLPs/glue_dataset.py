"""
gmlp/data/glue_dataset.py
-------------------------
GLUE (SST-2, MNLI) and SQuAD (v1.1, v2.0) datasets for NLP finetuning.

Paper Section 4.3–4.4: "Pay Attention to MLPs" (arXiv:2105.08050)

Finetuning tasks evaluated in the paper (Table 6):
  - SST-2  : single-sentence sentiment (GLUE). Metric: accuracy.
  - MNLI   : sentence-pair NLI, matched + mismatched splits. Metric: accuracy.
  - SQuAD v1.1 : extractive QA. Metric: F1.
  - SQuAD v2.0 : extractive QA with unanswerable questions. Metric: F1.

Hyperparameters from paper Table 9:
  - SST-2/MNLI: max_seq_len=128, batch={16,32}, lr∈{1e-5,2e-5,3e-5}, 5 epochs
  - SQuAD:      max_seq_len=512, batch=32,     lr=5e-5,               8K steps

Paper ref: Section 4.4, Tables 6 & 9
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SST2Dataset(Dataset):
    """
    Stanford Sentiment Treebank (binary classification).
    GLUE benchmark task SST-2.
    """

    LABEL_MAP = {"0": 0, "1": 1, "negative": 0, "positive": 1}

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_seq_len: int = 128,
        data_dir: Optional[str] = None,
    ) -> None:
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        raw = load_dataset("glue", "sst2", split=split, cache_dir=data_dir)
        self.examples = list(raw)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["sentence"],
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


class MNLIDataset(Dataset):
    """
    Multi-Genre Natural Language Inference.
    GLUE benchmark task MNLI (matched and mismatched).
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",          # 'train', 'validation_matched', 'validation_mismatched'
        max_seq_len: int = 128,
        data_dir: Optional[str] = None,
    ) -> None:
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        raw = load_dataset("glue", "mnli", split=split, cache_dir=data_dir)
        self.examples = list(raw)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ex = self.examples[idx]
        # Sentence-pair: premise + hypothesis
        # Paper note: cross-sentence alignment is where self-attention helps most (Sec 4.3)
        enc = self.tokenizer(
            ex["premise"],
            ex["hypothesis"],
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


class SQuADDataset(Dataset):
    """
    Stanford Question Answering Dataset (v1.1 or v2.0).
    Span-extraction QA: model predicts start/end token positions.

    Paper reports F1 on both SQuAD v1.1 and v2.0 (Table 6).
    gMLPlarge achieves 89.5 F1 on v1.1 without any self-attention.
    aMLPlarge achieves 92.2 / 85.4 F1 on v1.1/v2.0 (outperforms BERTlarge).
    """

    def __init__(
        self,
        tokenizer,
        version: str = "1.1",          # '1.1' or '2.0'
        split: str = "train",
        max_seq_len: int = 512,
        doc_stride: int = 128,
        data_dir: Optional[str] = None,
    ) -> None:
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        dataset_name = "squad" if version == "1.1" else "squad_v2"
        raw = load_dataset(dataset_name, split=split, cache_dir=data_dir)
        self.examples = list(raw)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["question"],
            ex["context"],
            max_length=self.max_seq_len,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Use first window only (simplified; production use sliding window)
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        # Compute start/end positions for the answer span
        start_pos = torch.tensor(0, dtype=torch.long)
        end_pos = torch.tensor(0, dtype=torch.long)
        if ex.get("answers") and ex["answers"]["answer_start"]:
            answer_start_char = ex["answers"]["answer_start"][0]
            answer_text = ex["answers"]["text"][0]
            answer_end_char = answer_start_char + len(answer_text)
            offsets = enc["offset_mapping"][0]
            for token_idx, (start, end) in enumerate(offsets.tolist()):
                if start <= answer_start_char < end:
                    start_pos = torch.tensor(token_idx, dtype=torch.long)
                if start < answer_end_char <= end:
                    end_pos = torch.tensor(token_idx, dtype=torch.long)
                    break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_pos,
            "end_positions": end_pos,
        }
