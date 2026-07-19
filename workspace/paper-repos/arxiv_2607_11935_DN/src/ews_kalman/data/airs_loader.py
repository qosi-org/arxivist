"""
NASA AIRS monthly regional data loader.

Implements Section 2.1 of arXiv:2607.11935: loads surface skin temperature
(T, variable SurfSkinTemp_A) and surface water-vapor mixing ratio (q,
variable H2O_MMR_Surf_A) for three regions (Arctic 65-90N, Tropics
10S-10N, Indian Monsoon 60E-100E/5N-25N), 284 monthly observations
(Sept 2002 - Apr 2026).

NASA AIRS data is freely available via the Giovanni portal
(https://giovanni.gsfc.nasa.gov/giovanni/), but requires either an
interactive download or NASA Earthdata API credentials -- see
data/README_data.md. A synthetic fallback generator is provided so the
pipeline runs end-to-end without live data access; it is calibrated only to
roughly match the paper's reported |beta| and sigma_beta magnitudes per
region (Table 1), not to reproduce the real AIRS observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


REGION_BOUNDS = {
    "arctic": {"lat_min": 65, "lat_max": 90, "lon_min": -180, "lon_max": 180},
    "tropics": {"lat_min": -10, "lat_max": 10, "lon_min": -180, "lon_max": 180},
    "monsoon": {"lat_min": 5, "lat_max": 25, "lon_min": 60, "lon_max": 100},
}

N_OBSERVATIONS = 284

_REGION_SEED_OFFSETS = {"arctic": 0, "tropics": 1, "monsoon": 2}


def _stable_region_offset(region_name: str) -> int:
    """Deterministic per-region seed offset.

    Python's built-in `hash()` is randomised per-process for strings unless
    PYTHONHASHSEED is fixed, so it must never be used for anything that
    needs to be reproducible across runs/machines. This lookup table is
    used instead.
    """
    return _REGION_SEED_OFFSETS.get(region_name, 0)


class AIRSDataLoader:
    """Loads (or synthetically generates) NASA AIRS regional T, q series."""

    def __repr__(self) -> str:  # noqa: D105
        return "AIRSDataLoader()"

    def region_bounds(self, region_name: str) -> Dict[str, float]:
        """Lat/lon bounding box for a named region.

        Args:
            region_name: one of 'arctic', 'tropics', 'monsoon'.

        Returns:
            Dict with lat_min, lat_max, lon_min, lon_max.

        Raises:
            ValueError: if region_name is not recognised.
        """
        if region_name not in REGION_BOUNDS:
            raise ValueError(f"Unknown region '{region_name}'; expected one of {list(REGION_BOUNDS)}")
        return REGION_BOUNDS[region_name]

    def load_region(
        self, region_name: str, data_dir: str = "data/raw", use_synthetic_fallback: bool = True, seed: int = 0
    ) -> Dict[str, np.ndarray]:
        """Load a region's T, q monthly series from a local CSV, or
        synthesize a plausible fallback series if no CSV is present.

        Expected CSV format (if using real data, see data/README_data.md):
            data/raw/airs_<region_name>.csv with columns: date, T, q

        Args:
            region_name: one of 'arctic', 'tropics', 'monsoon'.
            data_dir: directory to look for `airs_<region_name>.csv`.
            use_synthetic_fallback: if True and no CSV is found, generate a
                synthetic fallback series instead of raising.
            seed: seed for the synthetic fallback generator.

        Returns:
            Dict with 'dates' (pandas DatetimeIndex), 'T', 'q' (np.ndarray, shape [284]).

        Raises:
            FileNotFoundError: if no CSV is found and use_synthetic_fallback is False.
        """
        self.region_bounds(region_name)  # validates region_name

        csv_path = Path(data_dir) / f"airs_{region_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["date"])
            return {"dates": pd.DatetimeIndex(df["date"]), "T": df["T"].values, "q": df["q"].values}

        if not use_synthetic_fallback:
            raise FileNotFoundError(
                f"No AIRS data found at {csv_path}. Set use_synthetic_fallback=True "
                f"or populate data/raw/ per data/README_data.md."
            )

        return self._synthetic_fallback(region_name, seed=seed)

    def _synthetic_fallback(self, region_name: str, seed: int = 0) -> Dict[str, np.ndarray]:
        """Generate a plausible synthetic T, q series for a region.

        Calibrated only to roughly match the paper's reported per-region
        |beta| magnitude (Table 1): ~0.11 for Arctic (weak/noisy coupling),
        ~0.49 for Tropics (near Clausius-Clapeyron, stable), ~0.48 for
        Monsoon (near C-C but decelerating in the back half of the series).
        This is illustrative synthetic data, NOT real AIRS observations.
        """
        rng = np.random.default_rng(seed + _stable_region_offset(region_name))
        N = N_OBSERVATIONS
        t = np.arange(N)
        dates = pd.date_range("2002-09-01", periods=N, freq="MS")

        # Seasonal cycle + slow warming trend, region-specific means
        base_T = {"arctic": 250.0, "tropics": 300.0, "monsoon": 295.0}[region_name]
        seasonal_amp = {"arctic": 15.0, "tropics": 2.0, "monsoon": 5.0}[region_name]
        warming_trend = {"arctic": 0.03, "tropics": 0.01, "monsoon": 0.015}[region_name]

        T = (
            base_T
            + warming_trend * t
            + seasonal_amp * np.sin(2 * np.pi * t / 12)
            + rng.normal(0, 1.0, N)
        )

        if region_name == "arctic":
            # weak, intermittent coupling: beta drifts/regime-switches around ~0.1
            beta_true = 0.11 + 0.15 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 0.05, N)
        elif region_name == "tropics":
            # stable, near Clausius-Clapeyron ~0.5
            beta_true = 0.49 + rng.normal(0, 0.02, N)
        else:  # monsoon
            # near C-C but decelerating in the second half (coupling weakening)
            beta_true = 0.5 - 0.15 * (t / N) ** 2 + rng.normal(0, 0.04, N)

        log_q = beta_true * np.log(T) + rng.normal(0, 0.01, N)
        q = np.exp(log_q)

        return {"dates": dates, "T": T, "q": q}
