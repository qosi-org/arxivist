"""
training/trainer.py — Annual recursive refitting loop for Gu, Kelly, Xiu (2020).

Implements the training protocol from Section 2.1:
  "We refit once every year as most of our signals are updated annually.
   Each time we refit, we increase the training sample by 1 year."

The validation window rolls forward to include the most recent 12 months
at each refit. Hyperparameters are reselected each year from validation.

Paper reference: Section 2.1, Section 2.2 (recursive out-of-sample analysis)
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from asset_pricing_ml.data.dataset import PanelDataSplit, PanelSlice
from asset_pricing_ml.evaluation.metrics import ReturnMetrics
from asset_pricing_ml.utils.config import Config


class RecursiveTrainer:
    """Recursive annual refitting loop over the 30-year test period.

    Protocol (Section 2.1):
      - Training sample starts at 1957–1974; grows +1 year each refit
      - Validation = most recent 12 months of rolling window (1975–1986 initially)
      - Test = 1987–2016, evaluated one year at a time as it becomes available

    At each annual refit:
      1. Accumulate training data through year Y-1
      2. Use most recent 12 months as validation for hyperparameter selection
      3. Refit chosen model with best hyperparameters on full training set
      4. Generate out-of-sample predictions for year Y (test months)

    This produces 30 sets of test predictions covering the full 1987-2016 period.

    Args:
        cfg: Experiment config.
        model_factory: Callable(cfg) → new model instance (unfitted).
    """

    def __init__(self, cfg: Config, model_factory):
        self.cfg = cfg
        self.model_factory = model_factory
        self.fitted_models_: List[Any] = []
        self.test_predictions_: List[np.ndarray] = []
        self.test_actuals_: List[np.ndarray] = []
        self.test_mktcap_: List[np.ndarray] = []
        self.test_yearmonths_: List[str] = []

    def run(self, data: PanelDataSplit) -> Tuple[np.ndarray, np.ndarray]:
        """Execute the full recursive refitting loop.

        Returns:
            (all_predictions [NT_test], all_actuals [NT_test])
        """
        # Pool all months chronologically for the rolling window
        all_months = sorted(
            data.train + data.val + data.test,
            key=lambda s: s.year_month
        )

        # Group months by year
        from collections import defaultdict
        by_year: Dict[int, List[PanelSlice]] = defaultdict(list)
        for s in all_months:
            yr = int(s.year_month[:4])
            by_year[yr].append(s)

        val_window = self.cfg.data.val_window_years
        train_end = self.cfg.data.train_end_year
        test_start = self.cfg.data.val_end_year + 1
        test_end = self.cfg.data.test_end_year

        print(f"\nRecursive training: {test_end - test_start + 1} annual refits")
        print(f"  Model: {self.cfg.model.variant}")
        print(f"  Test period: {test_start}–{test_end}")

        for test_year in range(test_start, test_end + 1):
            t0 = time.time()

            # Training: all data before (test_year - val_window)
            train_cutoff = test_year - val_window - 1
            train_months: List[PanelSlice] = []
            for yr in range(1957, train_cutoff + 1):
                train_months.extend(by_year.get(yr, []))

            # Validation: most recent val_window years before test_year
            val_months: List[PanelSlice] = []
            for yr in range(test_year - val_window, test_year):
                val_months.extend(by_year.get(yr, []))

            # Test: current test year
            test_months: List[PanelSlice] = by_year.get(test_year, [])

            if not train_months or not val_months or not test_months:
                continue

            # Stack into arrays
            Z_train = np.concatenate([s.Z for s in train_months])
            R_train = np.concatenate([s.R for s in train_months])
            Z_val   = np.concatenate([s.Z for s in val_months])
            R_val   = np.concatenate([s.R for s in val_months])

            # Fit model with hyperparameter tuning
            model = self._fit_model(Z_train, R_train, Z_val, R_val)
            self.fitted_models_.append(model)

            # Generate predictions for each test month
            for s in test_months:
                pred = model.predict(s.Z)
                self.test_predictions_.append(pred)
                self.test_actuals_.append(s.R)
                self.test_mktcap_.append(s.mkt_cap)
                self.test_yearmonths_.append(s.year_month)

            # Compute validation R² for logging
            val_r2 = ReturnMetrics.oos_r2(R_val, model.predict(Z_val))
            elapsed = time.time() - t0
            print(
                f"  {test_year}: train_n={len(Z_train):,}, val_R²={val_r2*100:.3f}%, "
                f"test_months={len(test_months)}, time={elapsed:.1f}s"
            )

        all_pred = np.concatenate(self.test_predictions_)
        all_actual = np.concatenate(self.test_actuals_)
        return all_pred, all_actual

    def _fit_model(
        self,
        Z_train: np.ndarray, R_train: np.ndarray,
        Z_val: np.ndarray,   R_val: np.ndarray,
    ) -> Any:
        """Fit and tune model via validation sample."""
        variant = self.cfg.model.variant

        if variant in ("NN1", "NN2", "NN3", "NN4", "NN5"):
            return self._fit_nn(Z_train, R_train, Z_val, R_val)
        elif variant == "ENet":
            return self._fit_enet(Z_train, R_train, Z_val, R_val)
        elif variant == "PCR":
            return self._fit_pcr(Z_train, R_train, Z_val, R_val)
        elif variant == "PLS":
            return self._fit_pls(Z_train, R_train, Z_val, R_val)
        elif variant == "GLM":
            return self._fit_glm(Z_train, R_train, Z_val, R_val)
        elif variant == "RF":
            return self._fit_rf(Z_train, R_train, Z_val, R_val)
        elif variant == "GBRT":
            return self._fit_gbrt(Z_train, R_train, Z_val, R_val)
        elif variant == "OLS3":
            return self._fit_ols3(Z_train, R_train)
        elif variant == "OLS":
            return self._fit_ols(Z_train, R_train)
        else:
            raise ValueError(f"Unknown model variant: {variant}")

    def _fit_nn(self, Z_tr, R_tr, Z_vl, R_vl):
        from asset_pricing_ml.models.neural_net import NeuralNetModel
        layers = self.cfg.model.get_nn_layers()
        # Tune L1 lambda over grid via validation R²
        best_lambda, best_r2 = self.cfg.training.l1_lambda_grid[0], -np.inf
        for lam in self.cfg.training.l1_lambda_grid:
            model = NeuralNetModel(
                input_dim=Z_tr.shape[1], hidden_layers=layers,
                use_huber=self.cfg.model.use_huber_loss,
                n_ensemble_seeds=self.cfg.model.n_ensemble_seeds,
                lr=self.cfg.training.lr, l1_lambda=lam,
                batch_size=self.cfg.training.batch_size,
                early_stopping_patience=self.cfg.training.early_stopping_patience,
                max_epochs=self.cfg.training.max_epochs,
                device=self.cfg.hardware.device,
            )
            model.fit(Z_tr, R_tr, Z_vl, R_vl)
            r2 = ReturnMetrics.oos_r2(R_vl, model.predict(Z_vl))
            if r2 > best_r2:
                best_r2, best_lambda = r2, lam
                best_model = model
        return best_model

    def _fit_enet(self, Z_tr, R_tr, Z_vl, R_vl):
        from asset_pricing_ml.models.linear import ElasticNetModel
        m = ElasticNetModel()
        m.tune(Z_tr, R_tr, Z_vl, R_vl,
               self.cfg.training.__dict__.get("lambda_grid", [0.0001, 0.001, 0.01, 0.1, 1.0]),
               self.cfg.training.__dict__.get("rho_grid", [0.0, 0.25, 0.5, 0.75, 1.0]))
        return m

    def _fit_pcr(self, Z_tr, R_tr, Z_vl, R_vl):
        from asset_pricing_ml.models.linear import PCRModel
        m = PCRModel()
        m.tune(Z_tr, R_tr, Z_vl, R_vl, [1, 2, 5, 10, 20, 40])
        return m

    def _fit_pls(self, Z_tr, R_tr, Z_vl, R_vl):
        from asset_pricing_ml.models.linear import PLSModel
        m = PLSModel()
        m.tune(Z_tr, R_tr, Z_vl, R_vl, [1, 2, 3, 5, 10])
        return m

    def _fit_glm(self, Z_tr, R_tr, Z_vl, R_vl):
        from asset_pricing_ml.models.glm import GeneralizedLinearModel
        m = GeneralizedLinearModel()
        m.tune(Z_tr, R_tr, Z_vl, R_vl, [3, 4, 5], [0.001, 0.01, 0.1])
        return m

    def _fit_rf(self, Z_tr, R_tr, Z_vl, R_vl):
        from asset_pricing_ml.models.trees import RandomForestModel
        m = RandomForestModel()
        m.tune(Z_tr, R_tr, Z_vl, R_vl, [3, 5, 10, -1], ["sqrt", "third"])
        return m

    def _fit_gbrt(self, Z_tr, R_tr, Z_vl, R_vl):
        from asset_pricing_ml.models.trees import GBRTModel
        m = GBRTModel()
        m.tune(Z_tr, R_tr, Z_vl, R_vl, [1, 2, 3], [0.01, 0.05, 0.1], [500, 1000])
        return m

    def _fit_ols3(self, Z_tr, R_tr):
        from asset_pricing_ml.models.linear import OLSModel
        # OLS-3: use first 3 features (size, book-to-market, momentum)
        # as proxies for the Lewellen (2015) benchmark
        m = OLSModel(n_predictors=3)
        m.fit(Z_tr, R_tr)
        return m

    def _fit_ols(self, Z_tr, R_tr):
        from asset_pricing_ml.models.linear import OLSModel
        m = OLSModel()
        m.fit(Z_tr, R_tr)
        return m

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "test_predictions": self.test_predictions_,
                "test_actuals": self.test_actuals_,
                "test_mktcap": self.test_mktcap_,
                "test_yearmonths": self.test_yearmonths_,
                "config": self.cfg,
            }, f)

    @classmethod
    def load_results(cls, path: str) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)
