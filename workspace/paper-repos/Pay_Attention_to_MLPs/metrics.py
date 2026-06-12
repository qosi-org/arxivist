"""
gmlp/evaluation/metrics.py
--------------------------
Evaluation metrics for gMLP experiments.

Paper: "Pay Attention to MLPs" (arXiv:2105.08050)

Metrics by task:
  - MLM pretraining:    Validation perplexity = exp(mean cross-entropy)
  - SST-2 / MNLI:       Accuracy (%, matched and mismatched for MNLI)
  - SQuAD v1.1/v2.0:    F1 score over token spans
  - ImageNet:           Top-1 accuracy

Paper results reported as:
  - Mean ± std across 3 independent pretrain runs (std ≈ 0.01 perplexity)
  - Median of 5 independent finetuning runs (Table 6)
  - Mean ± std ≈ 0.1% for ImageNet Top-1

Paper ref: Section 4, Tables 2, 3, 4, 6
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Optional
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(losses: List[float]) -> float:
    """
    Compute perplexity from a list of per-batch cross-entropy losses.
    perplexity = exp(mean_loss)

    Paper: validation perplexity used as primary pretraining metric (Section 4.1).
    Reported std across runs ≈ 0.01 (Table 3 footnote).
    """
    import math
    avg_loss = sum(losses) / max(len(losses), 1)
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(preds: Tensor, labels: Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        preds:  [N] int64 — predicted class indices
        labels: [N] int64 — ground-truth class indices

    Returns:
        Accuracy as float in [0, 1].
    """
    assert preds.shape == labels.shape, \
        f"Shape mismatch: preds={preds.shape}, labels={labels.shape}"
    return (preds == labels).float().mean().item()


def compute_top1_accuracy(logits: Tensor, labels: Tensor) -> float:
    """
    ImageNet Top-1 accuracy from logits.

    Paper Table 2: gMLP-B achieves 81.6% Top-1 on ImageNet-1K.
    """
    preds = logits.argmax(dim=-1)
    return compute_accuracy(preds, labels)


# ---------------------------------------------------------------------------
# SQuAD F1 (token-level)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower case, remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1_tokens(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between predicted and gold answer strings."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_squad_f1(predictions: List[str], references: List[List[str]]) -> float:
    """
    Macro-average F1 over all SQuAD examples.
    Each example may have multiple reference answers (take max F1 over them).

    Paper Table 6:
      aMLPlarge achieves 92.2 / 85.4 F1 on SQuAD v1.1/v2.0 (outperforms BERTlarge).
    """
    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} predictions, {len(references)} references"
    f1_scores = []
    for pred, refs in zip(predictions, references):
        best_f1 = max(compute_f1_tokens(pred, ref) for ref in refs) if refs else 0.0
        f1_scores.append(best_f1)
    return sum(f1_scores) / max(len(f1_scores), 1)


def compute_exact_match(predictions: List[str], references: List[List[str]]) -> float:
    """Exact match: 1.0 if prediction matches any reference after normalisation."""
    scores = [
        float(any(normalize_answer(pred) == normalize_answer(ref) for ref in refs))
        for pred, refs in zip(predictions, references)
    ]
    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# Aggregation (median of N runs — paper protocol)
# ---------------------------------------------------------------------------

def aggregate_runs(metric_values: List[float]) -> dict:
    """
    Aggregate results from multiple runs per paper protocol.

    Paper Table 6 footnote: "each result entry was obtained by taking the
    median of five independent runs."

    Returns dict with median, mean, std, min, max.
    """
    import statistics
    return {
        "median": statistics.median(metric_values),
        "mean": statistics.mean(metric_values),
        "std": statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0,
        "min": min(metric_values),
        "max": max(metric_values),
        "n_runs": len(metric_values),
    }
