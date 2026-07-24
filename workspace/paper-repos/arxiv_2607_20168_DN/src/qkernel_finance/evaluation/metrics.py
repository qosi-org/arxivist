"""
Performance metrics (Sec 4.5, 5.2): mean IC, ICIR, t-stat, hit rate, paired
significance tests, and Holm family-wise correction.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


class PerformanceMetrics:
    """Computes the paper's window-level and summary metrics."""

    def rank_ic(self, predictions: np.ndarray, returns: np.ndarray) -> float:
        """Spearman rank IC for one rebalance date's cross-section."""
        if np.std(predictions) == 0 or np.std(returns) == 0:
            return 0.0
        ic, _p = spearmanr(predictions, returns)
        return float(ic) if not np.isnan(ic) else 0.0

    def summarize(self, ic_series: np.ndarray) -> dict:
        """
        Args:
            ic_series: [num_windows] per-window rank IC values.

        Returns:
            {"mean_ic":.., "icir":.., "t_stat":.., "hit_rate":..}
        """
        ic_series = np.asarray(ic_series)
        mean_ic = float(np.mean(ic_series))
        std_ic = float(np.std(ic_series, ddof=1)) if len(ic_series) > 1 else 0.0
        icir = mean_ic / std_ic if std_ic > 0 else float("nan")
        t_stat = icir * np.sqrt(len(ic_series)) if not np.isnan(icir) else float("nan")
        hit_rate = float(np.mean(ic_series > 0))
        return {"mean_ic": mean_ic, "icir": icir, "t_stat": t_stat, "hit_rate": hit_rate}

    def paired_significance(self, ic_a: np.ndarray, ic_b: np.ndarray) -> dict:
        """Paired t-test and Wilcoxon signed-rank test between two models' per-window IC series (Sec 5)."""
        ic_a, ic_b = np.asarray(ic_a), np.asarray(ic_b)
        assert ic_a.shape == ic_b.shape, "Both IC series must cover the same windows"
        t_stat, t_p = ttest_rel(ic_a, ic_b)
        diff = ic_a - ic_b
        if np.allclose(diff, 0):
            w_p = 1.0
        else:
            _w_stat, w_p = wilcoxon(ic_a, ic_b)
        return {"delta_ic": float(np.mean(ic_a - ic_b)), "paired_t_p": float(t_p), "wilcoxon_p": float(w_p)}

    def holm_correct(self, p_values: list[float]) -> list[float]:
        """Holm-Bonferroni family-wise correction (Sec 5.2)."""
        _reject, p_adjusted, _alpha_sidak, _alpha_bonf = multipletests(p_values, method="holm")
        return p_adjusted.tolist()

    def __repr__(self) -> str:  # noqa: D105
        return "PerformanceMetrics()"
