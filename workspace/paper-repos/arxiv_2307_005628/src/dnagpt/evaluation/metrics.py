"""Metrics for DNAGPT tasks.

GSR recognition -> acc/f1/mcc/precision/recall (Table S2-S4); mRNA abundance ->
r2 (Fig 3f); artificial genome generation -> Wasserstein distance + correlation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
)


def compute_metrics(preds: List[int], labels: List[int], metric: str) -> Dict[str, float]:
    """GSR classification metrics. Always returns accuracy; adds the named metric."""
    p = np.asarray(preds)
    y = np.asarray(labels)
    out: Dict[str, float] = {"accuracy": float((p == y).mean())}
    if metric == "accuracy":
        pass
    elif metric == "mcc":
        out["mcc"] = float(matthews_corrcoef(y, p))
    elif metric == "f1":
        out["f1"] = float(f1_score(y, p, average="binary", zero_division=0))
    else:
        raise ValueError(f"Unknown metric {metric!r}")
    # full GSR panel for reference (paper reports all five)
    out["f1"] = float(f1_score(y, p, average="binary", zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y, p)) if len(set(y)) > 1 else 0.0
    out["precision"] = float(precision_score(y, p, zero_division=0))
    out["recall"] = float(recall_score(y, p, zero_division=0))
    return out


def compute_r2(preds: List[float], targets: List[float]) -> Dict[str, float]:
    """Coefficient of determination for mRNA abundance regression."""
    return {"r2": float(r2_score(np.asarray(targets), np.asarray(preds)))}
