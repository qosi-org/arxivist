"""
gmlp/data/mlm_dataset.py
------------------------
Masked Language Modelling dataset for gMLP pretraining.

Paper Section 4: "Pay Attention to MLPs" (arXiv:2105.08050)

BERT-style masking protocol (Devlin et al. 2018, referenced in paper):
  - 15% of tokens are selected for masking
  - Of those: 80% → [MASK], 10% → random token, 10% → unchanged
  - Loss computed only on masked positions (labels=-100 elsewhere)

Tokenizer: paper uses 32K cased SentencePiece vocabulary.
Proxy used here: google/t5-base tokenizer (32K SentencePiece, cased).
See SIR implementation_assumptions[assume_007] and risk_assessment[R7].

Dataset: C4/English (full, ~300GB). Streaming mode recommended.
Ablation subset: C4/RealNews (smaller, used in Section 4.1 ablations).

Paper ref: Section 4, Table 8, SIR training_pipeline.nlp_pretraining
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset


class MLMDataset(IterableDataset):
    """
    Streaming MLM dataset wrapping HuggingFace C4.

    Yields batches of tokenized text with BERT-style masking applied.

    Args:
        tokenizer:       HuggingFace tokenizer (SentencePiece 32K cased proxy).
        max_seq_len:     Maximum sequence length. Paper: 512 (full), 128 (ablation).
        mlm_probability: Fraction of tokens to mask. Paper: 0.15 (BERT standard).
        dataset_name:    HuggingFace dataset name. Paper uses 'c4'.
        dataset_config:  Dataset subset. 'en' (full) or 'realnewslike' (ablation).
        split:           'train' or 'validation'.
        use_streaming:   Use HuggingFace streaming to avoid materialising full dataset.
                         Strongly recommended for C4/English (~300GB).
        data_dir:        Local cache directory.
        seed:            Random seed for masking.
    """

    MASK_TOKEN_PROB = 0.80    # of masked: replace with [MASK]
    RANDOM_TOKEN_PROB = 0.10  # of masked: replace with random token
    # Remaining 10%: keep original token

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 512,
        mlm_probability: float = 0.15,
        dataset_name: str = "c4",
        dataset_config: str = "en",
        split: str = "train",
        use_streaming: bool = True,
        data_dir: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        self.vocab_size = tokenizer.vocab_size
        self.seed = seed

        # Load dataset — streaming avoids full C4 materialisation (risk_001)
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=use_streaming,
            cache_dir=data_dir,
            trust_remote_code=True,
        )
        if use_streaming:
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10_000)

    def _tokenize_and_mask(self, text: str) -> Dict[str, Tensor]:
        """
        Tokenize a single text and apply BERT-style masking.

        Returns dict with:
          input_ids:      [max_seq_len] int64 — masked token ids
          labels:         [max_seq_len] int64 — original ids at masked pos, -100 elsewhere
          attention_mask: [max_seq_len] int64 — 1 for real tokens, 0 for padding
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)         # [max_seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_seq_len]

        # Apply BERT-style 15% masking
        input_ids, labels = self._mask_tokens(input_ids, attention_mask)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def _mask_tokens(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Apply BERT masking: 15% of non-padding tokens selected.
        Of those: 80% [MASK], 10% random, 10% unchanged.

        Paper ref: BERT protocol (Devlin et al. 2018), referenced in Section 4.
        SIR assume_003: masking not re-specified; standard BERT assumed.
        """
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # ignore padding in loss

        # Probability matrix: only mask real (non-padding) tokens
        prob_matrix = torch.full(input_ids.shape, self.mlm_probability)
        prob_matrix[attention_mask == 0] = 0.0

        # Skip special tokens (token ids 0–4 are typically special)
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        prob_matrix[special_tokens_mask] = 0.0

        # Select positions to mask
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100   # only compute loss on masked tokens

        # 80% → replace with [MASK] token
        mask_token_id = self.tokenizer.mask_token_id or 103  # fallback to BERT [MASK]=103
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, self.MASK_TOKEN_PROB)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = mask_token_id

        # 10% → replace with random token
        indices_random = (
            torch.bernoulli(
                torch.full(input_ids.shape, self.RANDOM_TOKEN_PROB / (1 - self.MASK_TOKEN_PROB))
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # Remaining 10%: keep original (labels already set above)
        return input_ids, labels

    def _get_special_tokens_mask(self, input_ids: Tensor) -> Tensor:
        """Identify special token positions (CLS, SEP, PAD) to exclude from masking."""
        special_ids = set()
        for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id"):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)
        mask = torch.zeros(input_ids.shape, dtype=torch.bool)
        for sid in special_ids:
            mask |= (input_ids == sid)
        return mask

    def __iter__(self):
        for example in self.dataset:
            text = example.get("text", "")
            if not text.strip():
                continue
            yield self._tokenize_and_mask(text)
