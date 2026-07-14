"""Downstream classification metrics: top-1 accuracy, MCC, F1.

Matches the metrics reported in HyenaDNA (GenomicBenchmarks: top-1 accuracy;
Nucleotide Transformer: MCC / F1).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef


def compute_metrics(preds: List[int], labels: List[int], which: List[str]) -> Dict[str, float]:
    """Compute requested metrics. Returns a dict keyed by metric name."""
    preds_a = np.asarray(preds)
    labels_a = np.asarray(labels)
    out: Dict[str, float] = {}
    if "accuracy" in which:
        out["accuracy"] = float((preds_a == labels_a).mean())
    if "mcc" in which:
        out["mcc"] = float(matthews_corrcoef(labels_a, preds_a))
    if "f1" in which:
        avg = "binary" if len(set(labels)) == 2 else "macro"
        out["f1"] = float(f1_score(labels_a, preds_a, average=avg))
    return out
