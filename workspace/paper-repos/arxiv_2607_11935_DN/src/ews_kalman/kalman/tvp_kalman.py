"""
TVP-Kalman forward filter and RTS backward smoother for time-varying
coupling-coefficient estimation.

Implements Section 2.2 of arXiv:2607.11935:

    log(q_k) = beta_k * log(T_k) + eps_k,   eps_k ~ N(0, R)      (observation eq.)
    x_k = F @ x_{k-1} + eta_k,               eta_k ~ N(0, Q)      (state eq.)
    x_k = [beta_k, beta'_k, beta''_k, beta'''_k]^T

with R = 1e-3, Q = diag(1e-6, 1e-7, 1e-8, 1e-9), dt = 1/12 year, followed by
Rauch-Tung-Striebel (RTS) backward smoothing.

This is a time-varying-parameter (TVP) regression via Kalman filter: the
regressor is treated as time-varying, i.e. the observation matrix at each
step is H_k = [x_reg_k, 0, 0, 0] (SIR implementation_assumption, confidence
0.75). The paper names the filter/smoother but does not spell out the
generic Kalman recursion equations themselves; the textbook forms are used
here.

Two observation modes are supported (see TVPKalmanFilter's `mode` arg):
"loglog" (the paper's real-data elasticity, Section 2.2) and "linear" (a
direct time-varying regression coefficient, needed because two of the six
simulated validation systems in Section 2.4 are explicitly defined via a
linear relationship y=beta(t)*x+eps, not a log-log power law).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ews_kalman.kalman.transition_matrix import build_taylor_transition_matrix


class TVPKalmanFilter:
    """Time-varying-parameter Kalman filter estimating a coupling
    coefficient beta(t) and its first three derivatives.

    Supports two observation modes:
      - "loglog": beta(t) = dlog(y)/dlog(x), the elasticity used for the
        real NASA AIRS T-q application (Section 2.2). Observation equation:
        log(y_k) = beta_k * log(x_k) + eps_k.
      - "linear": beta(t) as a direct time-varying linear regression
        coefficient, y_k = beta_k * x_k + eps_k. This matches how two of
        the six simulated validation systems are explicitly defined in the
        paper (Section 2.4: "beta step change" and "beta linear decay" both
        state y = beta(t)*x + eps with no logarithms), so applying the
        log-log elasticity filter to those two systems would be a
        methodological mismatch -- see architecture_plan.json risk_assessment.

    Args:
        R: observation noise variance (paper: 1e-3).
        Q_diag: process noise variances for [beta, beta', beta'', beta'''] (paper: [1e-6, 1e-7, 1e-8, 1e-9]).
        dt: time step in the same units as the derivatives are expressed (paper: 1/12 year for AIRS; 1.0 for simulations).
        order: highest tracked derivative (paper's 4-state vector => order=3).
        mode: "loglog" (default, matches Section 2.2) or "linear" (matches the two explicitly-linear simulated systems).
    """

    def __init__(
        self,
        R: float = 1e-3,
        Q_diag: Tuple[float, ...] = (1e-6, 1e-7, 1e-8, 1e-9),
        dt: float = 1.0 / 12.0,
        order: int = 3,
        mode: str = "loglog",
    ) -> None:
        if mode not in ("loglog", "linear"):
            raise ValueError(f"mode must be 'loglog' or 'linear', got {mode!r}")
        self.R = R
        self.Q = np.diag(Q_diag)
        self.dt = dt
        self.order = order
        self.mode = mode
        self.state_dim = order + 1
        self.F = build_taylor_transition_matrix(dt, order=order)

    def __repr__(self) -> str:  # noqa: D105
        return f"TVPKalmanFilter(R={self.R}, dt={self.dt}, state_dim={self.state_dim}, mode={self.mode!r})"

    def filter(
        self, x_reg: np.ndarray, y_obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the forward Kalman filter over an aligned pair of series.

        Args:
            x_reg: regressor series, shape [N]. In "loglog" mode this must
                already be log(x); in "linear" mode this is the raw
                regressor x.
            y_obs: response series, shape [N]. In "loglog" mode this must
                already be log(y); in "linear" mode this is the raw
                response y.

        Returns:
            (x_filt, P_filt, x_pred, P_pred): filtered state [N, state_dim],
            filtered covariance [N, state_dim, state_dim], one-step-ahead
            predicted state [N, state_dim], and predicted covariance
            [N, state_dim, state_dim] -- the latter two are required inputs
            to the RTS smoother.
        """
        N = len(x_reg)
        d = self.state_dim

        x_filt = np.zeros((N, d))
        P_filt = np.zeros((N, d, d))
        x_pred = np.zeros((N, d))
        P_pred = np.zeros((N, d, d))

        # Initial state: naive ratio-based guess for beta_0, derivatives at 0.
        x = np.zeros(d)
        x[0] = y_obs[0] / x_reg[0] if abs(x_reg[0]) > 1e-12 else 0.0
        P = np.eye(d) * 1.0  # diffuse-ish initial covariance

        for k in range(N):
            if k == 0:
                x_pr, P_pr = x, P
            else:
                x_pr = self.F @ x
                P_pr = self.F @ P @ self.F.T + self.Q

            x_pred[k] = x_pr
            P_pred[k] = P_pr

            H = np.zeros(d)
            H[0] = x_reg[k]

            y_resid = y_obs[k] - H @ x_pr
            S = H @ P_pr @ H.T + self.R
            K = (P_pr @ H) / S

            x = x_pr + K * y_resid
            P = P_pr - np.outer(K, H) @ P_pr

            x_filt[k] = x
            P_filt[k] = P

        return x_filt, P_filt, x_pred, P_pred

    def smooth(
        self, x_filt: np.ndarray, P_filt: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rauch-Tung-Striebel (RTS) backward smoothing pass.

        Args:
            x_filt, P_filt: forward-filtered state/covariance sequences.
            x_pred, P_pred: one-step-ahead predicted state/covariance
                sequences (both from `filter`).

        Returns:
            (x_smooth, P_smooth): smoothed state [N, state_dim] and
            covariance [N, state_dim, state_dim] sequences.
        """
        N, d = x_filt.shape
        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)

        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for k in range(N - 2, -1, -1):
            # Guard against a near-singular one-step-ahead covariance
            P_pred_next = P_pred[k + 1]
            try:
                C = P_filt[k] @ self.F.T @ np.linalg.inv(P_pred_next)
            except np.linalg.LinAlgError:
                C = P_filt[k] @ self.F.T @ np.linalg.pinv(P_pred_next)

            x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
            P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - P_pred_next) @ C.T

        return x_smooth, P_smooth

    def estimate_beta(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Convenience method: run filter + smooth on raw series and return
        the smoothed coupling coefficient and its derivatives.

        Args:
            x: raw regressor series (e.g. temperature T in loglog mode, or
                the raw driver in linear mode), shape [N]. In "loglog" mode
                all values must be > 0 (log is taken internally).
            y: raw response series (e.g. humidity q in loglog mode, or the
                raw response in linear mode), shape [N]. In "loglog" mode
                all values must be > 0.

        Returns:
            Dict with keys 'beta', 'beta_prime', 'beta_double_prime',
            'beta_triple_prime', each an array of shape [N].
        """
        if self.mode == "loglog":
            x_reg = np.log(x)
            y_obs = np.log(y)
        else:
            x_reg = np.asarray(x, dtype=float)
            y_obs = np.asarray(y, dtype=float)

        x_filt, P_filt, x_pred, P_pred = self.filter(x_reg, y_obs)
        x_smooth, _ = self.smooth(x_filt, P_filt, x_pred, P_pred)

        return {
            "beta": x_smooth[:, 0],
            "beta_prime": x_smooth[:, 1],
            "beta_double_prime": x_smooth[:, 2],
            "beta_triple_prime": x_smooth[:, 3],
        }
