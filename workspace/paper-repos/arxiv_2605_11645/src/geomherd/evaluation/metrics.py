"""
src/geomherd/evaluation/metrics.py
Detection and forecasting metrics for GeomHerd evaluation.
Paper: arXiv:2605.11645, Section 3.2

Implements:
  - Precision, Recall, FAR (Table 2/3)
  - AUROC, AUPRC
  - Conditional median lead time with 95% bootstrap CI
  - Paired-bootstrap lead difference (n_boot=5000, Table 2)
  - IQM MAE via rliable (Figure 4)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class DetectionMetrics:
    """
    Multi-axis detection evaluation following Section 3.2 and Tables 2-3.

    Paper reference: Section 3.2
        Metrics: precision, recall_super, FAR_sub, AUROC, AUPRC,
                 conditional median lead time, paired-bootstrap lead difference.
        Evaluation philosophy from Nikolopoulos (2025) for rare-event stability.
    """

    @staticmethod
    def precision_recall_far(
        alarm_times: List[Optional[int]],
        event_times: List[Optional[int]],
        is_supercritical: List[bool],
    ) -> Dict[str, float]:
        """
        Compute precision, supercritical recall, and subcritical FAR.

        Args:
            alarm_times: List of first alarm time per trajectory (None if no alarm)
            event_times: List of herding event time tau* per trajectory (None if no event)
            is_supercritical: Boolean mask (True = supercritical trajectory)
        Returns:
            Dict with keys: precision, recall_super, far_sub
        """
        assert len(alarm_times) == len(event_times) == len(is_supercritical)
        tp = fp = tn = fn = 0
        for alarm_t, event_t, is_super in zip(alarm_times, event_times, is_supercritical):
            fired = alarm_t is not None
            if is_super:
                # Event exists
                if fired and event_t is not None and alarm_t < event_t:
                    tp += 1  # correct early alarm
                elif fired:
                    fp += 1  # alarm but too late / event didn't happen
                else:
                    fn += 1  # missed event
            else:
                # Subcritical: no true event
                if fired:
                    fp += 1  # false alarm
                else:
                    tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_super = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far_sub = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # subcritical FAR
        return {"precision": precision, "recall_super": recall_super, "far_sub": far_sub}

    @staticmethod
    def auroc_auprc(
        scores: List[float],
        labels: List[int],
    ) -> Dict[str, float]:
        """
        Compute AUROC and AUPRC from continuous scores and binary labels.

        Args:
            scores: Continuous detection scores per trajectory
            labels: Binary labels (1=supercritical, 0=subcritical)
        """
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        try:
            auroc = float(roc_auc_score(labels_arr, scores_arr))
            auprc = float(average_precision_score(labels_arr, scores_arr))
        except ValueError:
            auroc = 0.5
            auprc = float(labels_arr.mean())
        return {"auroc": auroc, "auprc": auprc}

    @staticmethod
    def conditional_lead_time(
        alarm_times: List[Optional[int]],
        event_times: List[Optional[int]],
    ) -> Dict[str, float]:
        """
        Compute conditional median lead time and 95% bootstrap CI.

        Paper: Section 3.2 — conditional on both alarm and event firing before T.
        Lead = tau* - alarm_time (positive = alarm before event).

        Args:
            alarm_times: First alarm time per trajectory (None if no alarm)
            event_times: Herding event time per trajectory (None if no event)
        Returns:
            Dict with keys: median_lead, ci_lower, ci_upper, n_paired
        """
        leads = []
        for a_t, e_t in zip(alarm_times, event_times):
            if a_t is not None and e_t is not None:
                leads.append(e_t - a_t)
        if not leads:
            return {"median_lead": float("nan"), "ci_lower": float("nan"),
                    "ci_upper": float("nan"), "n_paired": 0}
        leads_arr = np.array(leads)
        median_lead = float(np.median(leads_arr))
        # Bootstrap CI
        n_boot = 5000
        rng = np.random.default_rng(0)
        boot_medians = []
        for _ in range(n_boot):
            sample = rng.choice(leads_arr, size=len(leads_arr), replace=True)
            boot_medians.append(np.median(sample))
        ci_lower = float(np.percentile(boot_medians, 2.5))
        ci_upper = float(np.percentile(boot_medians, 97.5))
        return {
            "median_lead": median_lead,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_paired": len(leads),
        }

    @staticmethod
    def paired_bootstrap_lead_diff(
        lead_A: List[int],
        lead_B: List[int],
        n_boot: int = 5000,
    ) -> Dict[str, float]:
        """
        Paired bootstrap for lead difference (GeomHerd - comparator).
        Paper: Section 3.3.1, Table 2 — n_boot=5000.

        Args:
            lead_A: Lead times for detector A (GeomHerd) on co-firing trajectories
            lead_B: Lead times for comparator B on same trajectories
        Returns:
            Dict with diff, ci_lower, ci_upper, p_value, n_paired
        """
        assert len(lead_A) == len(lead_B), "lead_A and lead_B must be same length"
        diffs = np.array(lead_A) - np.array(lead_B)
        median_diff = float(np.median(diffs))
        rng = np.random.default_rng(0)
        boot_medians = []
        for _ in range(n_boot):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            boot_medians.append(np.median(sample))
        ci_lower = float(np.percentile(boot_medians, 2.5))
        ci_upper = float(np.percentile(boot_medians, 97.5))
        # One-sided p-value: P(diff > 0) under null
        p_value = float(np.mean(np.array(boot_medians) <= 0))
        return {
            "lead_diff": median_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "n_paired": len(diffs),
        }


class ForecastMetrics:
    """
    Forecasting evaluation metrics for Kronos head (Section 3.3.3, Figure 4).

    Uses Interquartile Mean (IQM) MAE as implemented in rliable [41].
    """

    @staticmethod
    def iqm_mae(preds: np.ndarray, targets: np.ndarray) -> float:
        """
        IQM of absolute errors (25th–75th percentile mean).
        Paper: Section 3.3.3 — rliable IQM bars (Figure 4).
        """
        abs_errors = np.abs(preds - targets)
        q25, q75 = np.percentile(abs_errors, [25, 75])
        mask = (abs_errors >= q25) & (abs_errors <= q75)
        return float(abs_errors[mask].mean()) if mask.any() else float(abs_errors.mean())

    @staticmethod
    def cascade_window_mae(
        preds: np.ndarray,
        targets: np.ndarray,
        cascade_mask: np.ndarray,
    ) -> float:
        """
        MAE restricted to cascade window steps (where herding is active).
        Paper: Section 3.3.3 — cascade-window log-return MAE.
        """
        preds_cascade = preds[cascade_mask]
        targets_cascade = targets[cascade_mask]
        if len(preds_cascade) == 0:
            return float("nan")
        return ForecastMetrics.iqm_mae(preds_cascade, targets_cascade)
