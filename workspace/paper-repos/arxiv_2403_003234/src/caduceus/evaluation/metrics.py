"""Downstream metrics used across Caduceus's benchmarks.

Genomic Benchmarks -> top-1 accuracy (Table 1); NT tasks -> MCC / F1 (Table 2);
VEP -> AUROC (Fig 4).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score


def compute_metrics(preds: List[int], labels: List[int], metric: str) -> Dict[str, float]:
    """Compute the task metric plus accuracy for reference."""
    p = np.asarray(preds)
    y = np.asarray(labels)
    out: Dict[str, float] = {"accuracy": float((p == y).mean())}
    if metric == "accuracy":
        pass
    elif metric == "mcc":
        out["mcc"] = float(matthews_corrcoef(y, p))
    elif metric == "f1":
        out["f1"] = float(f1_score(y, p, average="macro"))
    elif metric == "auroc":
        out["auroc"] = float(roc_auc_score(y, p))
    else:
        raise ValueError(f"Unknown metric {metric!r}")
    return out
