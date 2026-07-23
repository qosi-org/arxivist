"""Downstream metrics used across GENERator's benchmarks.

Genomic Benchmarks -> accuracy (Table S5); NT tasks -> MCC (S3/S4);
Gener tasks -> weighted F1 (S6); VEP -> AUROC/AUPRC (S2).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_metrics(preds: List[int], labels: List[int], metric: str) -> Dict[str, float]:
    """Compute the task metric plus accuracy for reference."""
    p = np.asarray(preds)
    y = np.asarray(labels)
    out: Dict[str, float] = {"accuracy": float((p == y).mean())}
    if metric == "accuracy":
        pass
    elif metric == "mcc":
        out["mcc"] = float(matthews_corrcoef(y, p))
    elif metric == "weighted_f1":
        out["weighted_f1"] = float(f1_score(y, p, average="weighted"))
    else:
        raise ValueError(f"Unknown metric {metric!r}")
    return out


def compute_vep_metrics(scores: List[float], labels: List[int]) -> Dict[str, float]:
    """AUROC + AUPRC for variant effect prediction (label 1 = pathogenic)."""
    s = np.asarray(scores)
    y = np.asarray(labels)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }
