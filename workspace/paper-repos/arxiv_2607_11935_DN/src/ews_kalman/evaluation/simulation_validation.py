"""
Runs all six simulated tipping-point systems through both the TVP-Kalman
coupling-coefficient pipeline and the classical AR(1) pipeline, and reports
lead-time-to-tipping-point for each -- reproducing Table 3 of
arXiv:2607.11935.

Two of the six systems ("beta step change", "beta linear decay") are defined
in the paper via an explicit linear relationship y=beta(t)*x+eps (Section
2.4), so the Kalman filter is run in "linear" mode for those two. The
remaining four (fold bifurcation, logistic map, Stommel AMOC, critical
slowing down) have no explicit linear-in-beta structure; "loglog" mode
(matching the paper's real-data elasticity method, Section 2.2) is used for
these after shifting each series to be strictly positive.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ews_kalman.ews.classical_ews import ClassicalEWS
from ews_kalman.ews.lead_lag import LeadLagAnalyzer
from ews_kalman.kalman.tvp_kalman import TVPKalmanFilter
from ews_kalman.simulation.tipping_systems import TippingSystemSimulator

# Systems explicitly defined via a linear relationship y=beta(t)*x+eps in the
# paper (Section 2.4) -- use TVPKalmanFilter(mode="linear") for these.
_LINEAR_MODE_SYSTEMS = {"Beta step change", "Beta linear decay"}


class SimulationValidator:
    """Validates the beta-leads-AR(1) hypothesis on the six simulated
    tipping-point systems (Section 2.4, 3.4; Table 3, Figure 3).

    Args:
        kalman_kwargs: keyword arguments forwarded to TVPKalmanFilter
            (R, Q_diag, dt, order; `mode` is set automatically per system,
            see module docstring). Note: dt here is in simulation-timestep
            units (dt=1.0), not the paper's 1/12-year value used for the
            real AIRS analysis, since the simulated systems are unitless.
        ar1_window: rolling window length for the classical AR(1) comparison
            signal (kept short relative to simulation length, unlike the
            24-month window used for the real monthly AIRS series).
        burn_in: number of initial Kalman-filter timesteps excluded from
            both the baseline-statistics window and detection search, to
            avoid mistaking the filter's initial-state convergence
            transient for a genuine regime-change detection.
    """

    def __init__(
        self, kalman_kwargs: Optional[Dict] = None, ar1_window: int = 20, burn_in: int = 20
    ) -> None:
        default_kwargs = {"R": 1e-3, "Q_diag": (1e-4, 1e-5, 1e-6, 1e-7), "dt": 1.0}
        self.kalman_kwargs = {**default_kwargs, **(kalman_kwargs or {})}
        self.ar1_window = ar1_window
        self.burn_in = burn_in
        self._classical = ClassicalEWS()
        self._lead_lag = LeadLagAnalyzer()
        self._simulator = TippingSystemSimulator()

    def __repr__(self) -> str:  # noqa: D105
        return f"SimulationValidator(ar1_window={self.ar1_window}, burn_in={self.burn_in})"

    def _make_positive(self, series: np.ndarray) -> np.ndarray:
        """Shift a series to be strictly positive so log() is valid,
        preserving relative dynamics (needed since several simulated
        systems, e.g. critical slowing down, are centred near zero)."""
        min_val = np.min(series)
        if min_val <= 0:
            return series - min_val + 1.0
        return series

    def validate_one_system(
        self, system_name: str, x: np.ndarray, y: Optional[np.ndarray], tipping_index: int
    ) -> Dict:
        """Run one simulated system through both pipelines and compute lead times.

        Args:
            system_name: display name for this system (must match one of
                the six names used by validate_all_systems for correct
                linear-vs-loglog mode dispatch).
            x: primary (driver or univariate) series, shape [N].
            y: response series, shape [N], or None for univariate systems
                (fold bifurcation, critical slowing down), in which case a
                lag-1-shifted copy of x is used as the paired series so beta
                can still be estimated as a self-coupling coefficient.
            tipping_index: known tipping-point index.

        Returns:
            Dict with keys: simulation, tipping_t, beta_lead, ar1_lead, winner.
        """
        use_linear = system_name in _LINEAR_MODE_SYSTEMS

        if use_linear:
            x_in = np.asarray(x, dtype=float)
            y_in = np.roll(x_in, -1) if y is None else np.asarray(y, dtype=float)
            if y is None:
                y_in[-1] = y_in[-2]
            mode = "linear"
        else:
            x_in = self._make_positive(x)
            y_in = self._make_positive(np.roll(x_in, -1)) if y is None else self._make_positive(y)
            if y is None:
                y_in[-1] = y_in[-2]
            mode = "loglog"

        kf = TVPKalmanFilter(mode=mode, **self.kalman_kwargs)
        beta_result = kf.estimate_beta(x_in, y_in)
        beta = beta_result["beta"]

        ar1 = self._classical.rolling_ar1(x, window=self.ar1_window)
        # Pad AR(1) to align indices with the original series (rolling
        # window output is shorter than the input by window-1 samples)
        ar1_full = np.concatenate([np.full(self.ar1_window - 1, ar1[0]), ar1])

        # Baseline window for z-scoring excludes the Kalman filter's initial
        # convergence transient (burn_in) to avoid a spurious "detection" at
        # t=0 (see class docstring).
        baseline_start = self.burn_in
        baseline_end = max(baseline_start + 10, min(tipping_index // 2, baseline_start + 30))
        baseline_window = (baseline_start, baseline_end)

        beta_lead = self._lead_lag.simulation_lead_time(beta, tipping_index, baseline_window)
        ar1_lead = self._lead_lag.simulation_lead_time(ar1_full, tipping_index, baseline_window)

        if beta_lead is None and ar1_lead is None:
            winner = "neither"
        elif beta_lead is None:
            winner = "AR1"
        elif ar1_lead is None:
            winner = "beta"
        elif beta_lead > ar1_lead:
            winner = "beta"
        elif ar1_lead > beta_lead:
            winner = "AR1"
        else:
            winner = "tie"

        return {
            "simulation": system_name,
            "tipping_t": tipping_index,
            "beta_lead": beta_lead,
            "ar1_lead": ar1_lead,
            "winner": winner,
        }

    def validate_all_systems(self, sim_config: Optional[Dict] = None, seed: int = 0) -> List[Dict]:
        """Run all six simulated systems and assemble Table 3.

        Args:
            sim_config: optional dict of per-system parameter overrides,
                keyed by system name (matching config.yaml's
                `evaluation.simulation` block).
            seed: RNG seed forwarded to every simulator.

        Returns:
            List of 6 dicts, one per system, matching Table 3's columns.
        """
        cfg = sim_config or {}
        results = []

        fold = self._simulator.fold_bifurcation(seed=seed, **cfg.get("fold_bifurcation", {}))
        results.append(self.validate_one_system("Fold bifurcation", fold["x"], None, fold["tipping_index"]))

        step = self._simulator.beta_step_change(seed=seed, **cfg.get("beta_step_change", {}))
        results.append(
            self.validate_one_system("Beta step change", step["x"], step["y"], step["tipping_index"])
        )

        decay = self._simulator.beta_linear_decay(seed=seed, **cfg.get("beta_linear_decay", {}))
        results.append(
            self.validate_one_system("Beta linear decay", decay["x"], decay["y"], decay["tipping_index"])
        )

        logistic = self._simulator.logistic_map(seed=seed, **cfg.get("logistic_map", {}))
        results.append(
            self.validate_one_system("Logistic map", logistic["x"], None, logistic["tipping_index"])
        )

        amoc = self._simulator.stommel_amoc(seed=seed, **cfg.get("stommel_amoc", {}))
        results.append(
            self.validate_one_system("Stommel AMOC", amoc["T"], amoc["S"], amoc["tipping_index"])
        )

        csd = self._simulator.critical_slowing_down(seed=seed, **cfg.get("critical_slowing_down", {}))
        results.append(
            self.validate_one_system("Critical slowing down", csd["x"], None, csd["tipping_index"])
        )

        return results
