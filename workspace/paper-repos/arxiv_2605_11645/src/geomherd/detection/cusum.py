"""
src/geomherd/detection/cusum.py
CUSUM and Kendall-tau detection rules for herding and contagion alarms.
Paper: arXiv:2605.11645, Section 2.3 (Eqs. 4-6)

Implements:
  - One-sided Page CUSUM for kappa_bar_plus (Eq. 4)
  - One-sided Page CUSUM for beta_minus (Eq. 5)
  - Kendall-tau trend test on beta_minus (Eq. 6)
  - ContagionDetector: OR-combination of CUSUM + Kendall-tau (Eq. 6)
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np
from scipy.stats import kendalltau


class CUSUMDetector:
    """
    One-sided Page CUSUM detector (Page 1954) for upward mean shifts.

    Paper reference: Section 2.3, Eqs. 4 and 5
        S_t = max(0, S_{t-1} + (x_t - mu_base - k))
        A_t = 1[S_t > h]

    The baseline mean mu_base is estimated from the first `baseline_window`
    observations (pre-stress period). k is the allowance (slack); h is the threshold.

    Args:
        k: CUSUM allowance / slack parameter
        h: Detection threshold
        baseline_window: Number of initial samples used to estimate baseline mean
        skip_initial: Steps to skip before starting baseline estimation
    """

    def __init__(
        self,
        k: float = 2.0,
        h: float = 4.0,
        baseline_window: int = 35,
        skip_initial: int = 0,
    ):
        self.k = k
        self.h = h
        self.baseline_window = baseline_window
        self.skip_initial = skip_initial
        self._S: float = 0.0
        self._n: int = 0
        self._baseline_samples: list = []
        self._mu_base: Optional[float] = None
        self._alarm_fired: bool = False

    def _estimate_baseline(self) -> None:
        if len(self._baseline_samples) >= self.baseline_window:
            self._mu_base = float(np.mean(self._baseline_samples[:self.baseline_window]))

    def update(self, value: float) -> Tuple[float, bool]:
        """
        Process one new observation.

        Args:
            value: New scalar observation (e.g. kappa_bar_plus or beta_minus)
        Returns:
            (S_t, alarm_fired): Current CUSUM statistic and alarm boolean
        """
        self._n += 1
        if self._n <= self.skip_initial:
            return self._S, False

        # Collect baseline samples
        if self._mu_base is None:
            self._baseline_samples.append(value)
            self._estimate_baseline()
            return self._S, False

        # Eq. 4 / Eq. 5: S_t = max(0, S_{t-1} + (x_t - mu_base - k))
        self._S = max(0.0, self._S + (value - self._mu_base - self.k))
        alarm = self._S > self.h
        if alarm:
            self._alarm_fired = True
        return self._S, alarm

    def reset(self) -> None:
        """Reset detector state (use between trajectories)."""
        self._S = 0.0
        self._n = 0
        self._baseline_samples = []
        self._mu_base = None
        self._alarm_fired = False

    def set_baseline(self, mu_base: float) -> None:
        """Manually set baseline mean (e.g. from pre-calibrated value)."""
        self._mu_base = mu_base

    @property
    def statistic(self) -> float:
        return self._S

    @property
    def baseline_mean(self) -> Optional[float]:
        return self._mu_base

    def __repr__(self) -> str:
        return (f"CUSUMDetector(k={self.k}, h={self.h}, "
                f"baseline_window={self.baseline_window}, S={self._S:.4f}, "
                f"mu_base={self._mu_base})")


class KendallTauDetector:
    """
    Rolling Kendall-tau trend test on a scalar time series.

    Paper reference: Section 2.3, Eq. 6 (complementary trend channel)
        A_t^{-,tau} = 1[Kendall_{[t-W_tau, t]}(beta_minus) > tau_thresh]

    Fires alarm when Kendall-tau rank correlation (trend strength) exceeds
    tau_thresh over the rolling window W_tau.

    # ASSUMED: tau_thresh = -0.4 (inferred from Table 3 'tau_neg=-0.4' label)
    # ASSUMED: W_tau = 20 (not stated in paper)
    # TODO: verify tau_thresh and W_tau from paper or authors.

    Args:
        tau_thresh: Kendall-tau threshold for alarm (ASSUMED: -0.4)
        window: Rolling window length W_tau (ASSUMED: 20)
    """

    def __init__(self, tau_thresh: float = -0.4, window: int = 20):
        self.tau_thresh = tau_thresh
        self.window = window
        self._buffer: deque = deque(maxlen=window)

    def update(self, value: float) -> Tuple[float, bool]:
        """
        Process one new observation and test for trend.

        Args:
            value: New scalar observation (e.g. beta_minus)
        Returns:
            (tau, alarm_fired): Kendall-tau statistic and alarm boolean
        """
        self._buffer.append(value)
        if len(self._buffer) < 4:  # Need minimum samples for rank correlation
            return 0.0, False
        x = np.array(self._buffer)
        t_idx = np.arange(len(x))
        tau, _ = kendalltau(t_idx, x)
        alarm = tau > self.tau_thresh
        return float(tau), alarm

    def reset(self) -> None:
        self._buffer.clear()

    def __repr__(self) -> str:
        return (f"KendallTauDetector(tau_thresh={self.tau_thresh}, "
                f"window={self.window}, buffer_len={len(self._buffer)})")


class ContagionDetector:
    """
    OR-combination of CUSUM and Kendall-tau for beta_minus contagion detection.

    Paper reference: Section 2.3, Eq. 6
        A_t^- = A_t^{-,cusum} OR 1[Kendall_{[t-W_tau,t]}(beta_minus) > tau_thresh]

    The CUSUM tracks the level of bridge emergence while Kendall-tau tracks
    its trend; the OR combination is robust to either signal alone being noisy.

    Args:
        cusum: CUSUMDetector for beta_minus
        kendall: KendallTauDetector for beta_minus trend
    """

    def __init__(
        self,
        cusum: Optional[CUSUMDetector] = None,
        kendall: Optional[KendallTauDetector] = None,
    ):
        # ASSUMED parameters for beta_minus CUSUM (not specified; using same h as kappa+)
        self.cusum = cusum or CUSUMDetector(k=0.5, h=4.0, baseline_window=35)
        self.kendall = kendall or KendallTauDetector(tau_thresh=-0.4, window=20)

    def update(self, beta_minus: float) -> Tuple[float, float, bool]:
        """
        Process one new beta_minus observation.

        Args:
            beta_minus: Fraction of bridge edges (contagion signal)
        Returns:
            (cusum_S, kendall_tau, alarm_fired): Statistics and OR-combined alarm
        """
        S, cusum_alarm = self.cusum.update(beta_minus)
        tau, kendall_alarm = self.kendall.update(beta_minus)
        # Eq. 6: A_t^- = A_t^{-,cusum} OR A_t^{-,kendall}
        alarm = cusum_alarm or kendall_alarm
        return S, tau, alarm

    def reset(self) -> None:
        self.cusum.reset()
        self.kendall.reset()

    def __repr__(self) -> str:
        return f"ContagionDetector(cusum={self.cusum}, kendall={self.kendall})"


class HerdingDetector:
    """
    CUSUM detector for kappa_bar_plus (herding / within-clique signal).

    Paper reference: Section 2.3, Eq. 4
        S_t^+ = max(0, S_{t-1}^+ + (kappa_bar_plus(t) - mu_base^+ - k_+))
        A_t^+ = 1[S_t^+ > h_+]

    Two calibrated operating points (Appendix D / Table 8):
        recall_oriented:    (k=0.5, h=4.0) -> 272-step median lead, recall=0.52
        precision_oriented: (k=2.0, h=4.0) -> 178-step median lead, recall=0.04
    """

    def __init__(
        self,
        operating_point: str = "precision",
        baseline_window: int = 35,
        skip_initial: int = 50,
    ):
        assert operating_point in ("recall", "precision"), \
            f"operating_point must be 'recall' or 'precision', got {operating_point}"
        if operating_point == "recall":
            k, h = 0.5, 4.0
        else:
            k, h = 2.0, 4.0
        self.operating_point = operating_point
        self._cusum = CUSUMDetector(
            k=k, h=h, baseline_window=baseline_window, skip_initial=skip_initial
        )

    def update(self, kappa_bar_plus: float) -> Tuple[float, bool]:
        """
        Process one kappa_bar_plus observation.
        Returns (S_t, alarm_fired).
        """
        return self._cusum.update(kappa_bar_plus)

    def reset(self) -> None:
        self._cusum.reset()

    def set_k_h(self, k: float, h: float) -> None:
        """Override k and h for custom operating points."""
        self._cusum.k = k
        self._cusum.h = h

    def __repr__(self) -> str:
        return (f"HerdingDetector(op={self.operating_point}, "
                f"k={self._cusum.k}, h={self._cusum.h})")
