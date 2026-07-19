"""GUE metrics: MCC (most tasks) and F1 (Covid variant).

Follows DNABERT-2 Table 12 metric assignment.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef


def compute_metrics(preds: List[int], labels: List[int], metric: str) -> Dict[str, float]:
    """Compute the GUE metric for a task. Returns a dict with the metric + accuracy."""
    p = np.asarray(preds)
    y = np.asarray(labels)
    out: Dict[str, float] = {"accuracy": float((p == y).mean())}
    if metric == "mcc":
        out["mcc"] = float(matthews_corrcoef(y, p))
    elif metric == "f1":
        avg = "binary" if len(set(labels)) == 2 else "macro"
        out["f1"] = float(f1_score(y, p, average=avg))
    else:
        raise ValueError(f"Unknown metric {metric!r} (expected 'mcc' or 'f1')")
    return out
