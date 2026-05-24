"""
data/dataset.py — Data loading, splitting, and synthetic generation.

Paper: Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning"
       Review of Financial Studies 33, 2223-2273. doi:10.1093/rfs/hhaa009

Data used in paper:
  - CRSP monthly stock returns: NYSE/AMEX/NASDAQ, 1957-03 to 2016-12
  - 94 firm characteristics from Green, Hand, Zhang (2017)
  - 8 macro predictors from Welch and Goyal (2008)
  - 74 industry dummies (first two SIC digits)

⚠ PROPRIETARY DATA: CRSP and characteristics require WRDS subscription.
  See data/README_data.md for instructions.
  Use SyntheticDataGenerator for development without real data.

Paper reference: Section 2.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from asset_pricing_ml.data.features import FeatureBuilder
from asset_pricing_ml.utils.config import DataConfig


@dataclass
class PanelSlice:
    """One time-period slice of the panel: features Z and returns R for N_t stocks."""
    year_month: str       # e.g. "1987-01"
    Z: np.ndarray         # [N_t, 920] float32 — feature matrix
    R: np.ndarray         # [N_t]      float32 — excess returns
    mkt_cap: np.ndarray   # [N_t]      float32 — market cap for value weighting
    permno: np.ndarray    # [N_t]      int     — CRSP stock identifier


@dataclass
class PanelDataSplit:
    """Temporal train/validation/test split maintaining time ordering.

    Paper Section 2.1:
      Training:   1957-1974 (18 years, grows +1 year at each annual refit)
      Validation: 1975-1986 (12-year rolling window)
      Test:       1987-2016 (30 years, never used for estimation or tuning)
    """
    train: List[PanelSlice]
    val: List[PanelSlice]
    test: List[PanelSlice]

    @property
    def n_train(self) -> int:
        return sum(s.Z.shape[0] for s in self.train)

    @property
    def n_val(self) -> int:
        return sum(s.Z.shape[0] for s in self.val)

    @property
    def n_test(self) -> int:
        return sum(s.Z.shape[0] for s in self.test)

    def stack_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stack all training slices into (Z [NT_train, 920], R [NT_train])."""
        Z = np.concatenate([s.Z for s in self.train], axis=0)
        R = np.concatenate([s.R for s in self.train], axis=0)
        return Z, R

    def stack_val(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stack all validation slices into (Z, R)."""
        Z = np.concatenate([s.Z for s in self.val], axis=0)
        R = np.concatenate([s.R for s in self.val], axis=0)
        return Z, R

    def stack_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stack all test slices into (Z, R)."""
        Z = np.concatenate([s.Z for s in self.test], axis=0)
        R = np.concatenate([s.R for s in self.test], axis=0)
        return Z, R

    def __repr__(self) -> str:
        return (
            f"PanelDataSplit(train={len(self.train)} months/{self.n_train:,} obs, "
            f"val={len(self.val)} months/{self.n_val:,} obs, "
            f"test={len(self.test)} months/{self.n_test:,} obs)"
        )


class StockReturnDataset:
    """Loads and preprocesses the Gu-Kelly-Xiu panel dataset.

    Builds the full 920-dimensional feature vector z_it = x_t ⊗ c_it
    from CRSP returns, characteristics, and macro predictors.

    Paper reference: Section 2.1

    Args:
        cfg: DataConfig with paths and split years.
        feature_builder: FeatureBuilder instance.
    """

    MACRO_VARS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]

    def __init__(self, cfg: DataConfig, feature_builder: Optional[FeatureBuilder] = None):
        self.cfg = cfg
        self.fb = feature_builder or FeatureBuilder()

    def load(self) -> PanelDataSplit:
        """Load all data and return a train/val/test split.

        Returns:
            PanelDataSplit with monthly slices.

        Raises:
            FileNotFoundError: If CRSP or characteristics data not found.
                See data/README_data.md for acquisition instructions.
        """
        import os
        for path in [self.cfg.crsp_path, self.cfg.chars_path, self.cfg.macro_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Data file not found: {path}\n"
                    "Real data requires WRDS subscription. "
                    "Use SyntheticDataGenerator for development — see data/README_data.md."
                )

        returns_df = pd.read_parquet(self.cfg.crsp_path)
        chars_df = pd.read_parquet(self.cfg.chars_path)
        macro_df = pd.read_csv(self.cfg.macro_path, index_col=0, parse_dates=True)

        slices = self._build_panel_slices(returns_df, chars_df, macro_df)

        train_slices = [s for s in slices if int(s.year_month[:4]) <= self.cfg.train_end_year]
        val_slices = [
            s for s in slices
            if self.cfg.train_end_year < int(s.year_month[:4]) <= self.cfg.val_end_year
        ]
        test_slices = [s for s in slices if int(s.year_month[:4]) > self.cfg.val_end_year]

        return PanelDataSplit(train=train_slices, val=val_slices, test=test_slices)

    def _build_panel_slices(
        self,
        returns_df: pd.DataFrame,
        chars_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> List[PanelSlice]:
        """Build per-month PanelSlice objects from raw data."""
        slices = []
        for ym, grp in returns_df.groupby("year_month"):
            year = int(str(ym)[:4])
            if year < 1957 or year > 2016:
                continue

            c_it = grp[chars_df.columns.intersection(grp.columns)].values
            x_t = macro_df.loc[str(ym), self.MACRO_VARS].values.astype(float)
            sic2 = grp["sic2"].values if "sic2" in grp.columns else np.zeros(len(grp), dtype=int)
            industry_dummies = self._encode_industry(sic2)

            Z = self.fb.build_full_feature_vector(c_it, x_t, industry_dummies)
            R = grp["excess_ret"].values.astype(np.float32)
            mkt_cap = grp["mkt_cap"].values.astype(np.float32) if "mkt_cap" in grp.columns else np.ones(len(grp), np.float32)
            permno = grp["permno"].values.astype(int) if "permno" in grp.columns else np.arange(len(grp))

            slices.append(PanelSlice(year_month=str(ym), Z=Z, R=R, mkt_cap=mkt_cap, permno=permno))

        return sorted(slices, key=lambda s: s.year_month)

    def _encode_industry(self, sic2: np.ndarray) -> np.ndarray:
        """One-hot encode 2-digit SIC industry codes into 74 dummies."""
        # Paper: 74 industry sector dummies based on first two SIC digits
        unique_sic = sorted(set(sic2))[:74]
        sic_to_idx = {s: i for i, s in enumerate(unique_sic)}
        dummies = np.zeros((len(sic2), 74), dtype=np.float32)
        for i, s in enumerate(sic2):
            if s in sic_to_idx:
                dummies[i, sic_to_idx[s]] = 1.0
        return dummies

    def __repr__(self) -> str:
        return f"StockReturnDataset(train_end={self.cfg.train_end_year}, test_end={self.cfg.test_end_year})"


class SyntheticDataGenerator:
    """Generate synthetic stock return panel data for development and testing.

    Simulates a factor model with stock characteristics as factor loadings:
        r_it+1 = beta_it' @ f_t + eps_it
        beta_it = theta @ c_it  (characteristics as loadings)

    This is NOT a faithful simulation of the real CRSP data. Results from
    synthetic data will NOT match the paper's reported numbers.

    Use this when WRDS/CRSP data is not available.

    Args:
        n_stocks: Number of stocks to simulate.
        n_months: Number of months to simulate.
        n_factors: Number of latent factors.
        seed: Random seed.
    """

    def __init__(
        self,
        n_stocks: int = 500,
        n_months: int = 720,   # 60 years × 12
        n_factors: int = 5,
        seed: int = 42,
    ):
        self.n_stocks = n_stocks
        self.n_months = n_months
        self.n_factors = n_factors
        self.rng = np.random.default_rng(seed)
        self.fb = FeatureBuilder()

    def generate(self) -> PanelDataSplit:
        """Generate a full panel split (train/val/test) with synthetic data.

        Returns:
            PanelDataSplit with 720 monthly slices split at months 216/360.
            (Corresponds to 18/12/30 year split for 60-year panel)
        """
        # Latent factor returns [n_months, n_factors]
        factor_returns = self.rng.standard_normal((self.n_months, self.n_factors)) * 0.02

        # Stock factor loadings [n_stocks, n_factors] — persistent with slow drift
        loadings = self.rng.standard_normal((self.n_stocks, self.n_factors)) * 0.5

        slices = []
        start_year, start_month = 1957, 3
        for t in range(self.n_months):
            # Vary number of active stocks slightly each month
            n_active = self.n_stocks + self.rng.integers(-20, 20)
            n_active = max(10, min(n_active, self.n_stocks))
            idx = self.rng.choice(self.n_stocks, size=n_active, replace=False)

            # Synthetic 94 characteristics [n_active, 94]
            c_it = self.rng.standard_normal((n_active, 94))

            # Synthetic macro [8]
            x_t = self.rng.standard_normal(8) * 0.5

            # Synthetic industry dummies [n_active, 74]
            sic_assignments = self.rng.integers(0, 74, size=n_active)
            industry_dummies = np.zeros((n_active, 74), dtype=np.float32)
            industry_dummies[np.arange(n_active), sic_assignments] = 1.0

            # Build feature vector
            Z = self.fb.build_full_feature_vector(c_it, x_t, industry_dummies)

            # Excess returns: factor model + noise
            beta = loadings[idx]  # [n_active, n_factors]
            R = (beta @ factor_returns[t]).astype(np.float32) + \
                self.rng.standard_normal(n_active).astype(np.float32) * 0.08

            mkt_cap = np.abs(self.rng.standard_normal(n_active)).astype(np.float32) * 1e9 + 1e7
            permno = idx.astype(int)

            year = start_year + (start_month - 1 + t) // 12
            month = (start_month - 1 + t) % 12 + 1
            ym = f"{year}-{month:02d}"

            slices.append(PanelSlice(year_month=ym, Z=Z, R=R, mkt_cap=mkt_cap, permno=permno))

        # Split proportional to paper ratio 18/12/30 years (60yr total)
        n = len(slices)
        train_end = int(n * 18 / 60)
        val_end   = int(n * (18 + 12) / 60)
        train = slices[:train_end]
        val   = slices[train_end:val_end]
        test  = slices[val_end:]
        return PanelDataSplit(train=train, val=val, test=test)

    def __repr__(self) -> str:
        return (
            f"SyntheticDataGenerator(n_stocks={self.n_stocks}, "
            f"n_months={self.n_months}, n_factors={self.n_factors})"
        )
