"""
Out-of-sample evaluation metrics.

Implements the paper's *corrected* OOS R2 (Appendix E — the v3 revision fixes an
error in scikit-learn's `r2_score()` denominator alignment) and the Diebold-Mariano
test with a HAC (Newey-West) estimator for serial-correlation-robust model comparison
(mentioned in the v3 revision notes and Table 4/6).

Corrected OOS R2 (Campbell & Thompson 2008 convention, Appendix E):
    R2_OOS = 1 - sum((y_i - yhat_i)^2) / sum((y_i - ybar_train_val)^2)
where ybar_train_val is the historical mean computed over BOTH training and
validation samples (not the OOS mean), per the v3 correction.
"""
from __future__ import annotations

import numpy as np
from statsmodels.stats.sandwich_covariance import cov_hac
import statsmodels.api as sm


class OOSMetrics:
    """Out-of-sample R2, MSE, and Diebold-Mariano HAC test (paper's corrected v3 methodology)."""

    @staticmethod
    def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_hist_mean: float) -> float:
        """Corrected out-of-sample R2 (Appendix E of the v3 paper revision).

        Args:
            y_true: [T] actual OOS returns for one stock.
            y_pred: [T] predicted OOS returns for one stock.
            y_hist_mean: scalar historical mean return computed over BOTH the
                training and validation samples (per Campbell & Thompson 2008,
                as corrected in the paper's v3 revision).

        Returns:
            Scalar OOS R2. Can be negative if the model underperforms the
            historical-mean benchmark.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - y_hist_mean) ** 2)
        if denominator == 0:
            return float("nan")
        return 1.0 - numerator / denominator

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def diebold_mariano_hac(
        errors_a: np.ndarray, errors_b: np.ndarray, lags: int = 12
    ) -> tuple[float, float]:
        """Diebold-Mariano test with a HAC (Newey-West) estimator (paper v3 revision notes).

        Tests whether model A's forecast errors are significantly different from
        model B's, robust to serial correlation in the loss-differential series.
        A negative DM statistic indicates model A (column) outperforms model B (row),
        matching the sign convention used in the paper's Table 4/6.

        Args:
            errors_a: [T] forecast errors (y_true - y_pred) for model A.
            errors_b: [T] forecast errors (y_true - y_pred) for model B.
            lags: HAC lag length. ASSUMED=12 (monthly data, ~1yr), config-tunable
                via evaluation.dm_test_lags_hac (SIR confidence 0.5).

        Returns:
            (dm_statistic, p_value)
        """
        errors_a = np.asarray(errors_a, dtype=np.float64)
        errors_b = np.asarray(errors_b, dtype=np.float64)
        assert errors_a.shape == errors_b.shape, "errors_a and errors_b must have the same shape"

        d = errors_a ** 2 - errors_b ** 2  # squared-error loss differential
        n = len(d)
        if n < 2:
            return float("nan"), float("nan")

        d_mean = d.mean()
        x = sm.add_constant(np.ones(n))
        model = sm.OLS(d, x).fit()
        hac_cov = cov_hac(model, nlags=min(lags, n - 1))
        se = np.sqrt(hac_cov[0, 0])
        if se == 0 or np.isnan(se):
            return float("nan"), float("nan")

        dm_stat = d_mean / se
        # Two-sided p-value using the normal approximation (paper reports significance stars).
        from scipy.stats import norm

        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
        return float(dm_stat), float(p_value)

    def __repr__(self) -> str:
        return "OOSMetrics()"
