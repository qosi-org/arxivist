"""Data preprocessing utilities for Quantum-SMOTE.

Implements BasePreprocessor and TelcoChurnPreprocessor per architecture plan and SIR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """Abstract base preprocessor interface."""

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class TelcoChurnPreprocessor(BasePreprocessor):
    """Preprocessor for the Telco Customer Churn dataset.

    Steps implemented (per SIR):
    - Drop rows with missing values (paper drops 11 rows)
    - Drop identifier columns (customerID)
    - Convert specified categorical columns to 'category' dtype
    - Convert TotalCharges, tenure, MonthlyCharges to float
    - One-hot encode categorical features (handle_unknown='ignore')
    - MinMax scale continuous features to [0,1]
    - Correlation-based feature selection to reduce to target_dim (F=32)
    """

    drop_columns: List[str] = None
    target_column: str = "Churn"
    correlation_threshold: float = 0.9
    target_dim: int = 32

    # fitted attributes
    ohe: Optional[OneHotEncoder] = None
    scaler: Optional[MinMaxScaler] = None
    label_encoder: Optional[LabelEncoder] = None
    feature_names_: Optional[List[str]] = None
    selected_feature_names_: Optional[List[str]] = None

    def __post_init__(self):
        if self.drop_columns is None:
            self.drop_columns = ["customerID"]

    def _ensure_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert well-known categorical columns to category dtype when present
        categorical_cols = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
        ]

        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Convert numeric columns
        for num_col in ["TotalCharges", "tenure", "MonthlyCharges"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        return df

    def _one_hot_encode(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        # Identify categorical columns (object or category), excluding target
        cat_cols = [
            c for c in df.columns if c != self.target_column and pd.api.types.is_categorical_dtype(df[c])
        ]

        # If none, return empty
        if not cat_cols:
            return np.zeros((len(df), 0)), []

        try:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            # Compatibility path for older scikit-learn releases
            self.ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        cat_arr = self.ohe.fit_transform(df[cat_cols])

        # Build feature names for one-hot columns
        categories = self.ohe.categories_
        ohe_feature_names = []
        for col, cats in zip(cat_cols, categories):
            for cat in cats:
                ohe_feature_names.append(f"{col}={cat}")

        return cat_arr, ohe_feature_names

    def _select_by_correlation(self, X_df: pd.DataFrame) -> List[str]:
        # Iteratively drop features with pairwise abs(corr) > threshold until <= target_dim
        df = X_df.copy()
        if df.shape[1] <= self.target_dim:
            return list(df.columns)

        corr = df.corr().abs()
        # Mask diagonal
        corr.values[[np.arange(corr.shape[0])] * 2] = 0

        while True:
            # find feature pairs exceeding threshold
            cols = corr.columns
            pairs = np.where(corr.values > self.correlation_threshold)
            if len(pairs[0]) == 0:
                break

            # compute mean absolute corr per column
            mean_abs = corr.mean().sort_values(ascending=False)
            # drop the feature with highest mean_abs
            drop_col = mean_abs.index[0]
            logger.debug("Dropping correlated feature %s (mean_abs_corr=%.3f)", drop_col, mean_abs.iloc[0])
            df = df.drop(columns=[drop_col])
            if df.shape[1] <= self.target_dim:
                break
            corr = df.corr().abs()
            corr.values[[np.arange(corr.shape[0])] * 2] = 0

        # If still too many features, fallback to variance selection
        if df.shape[1] > self.target_dim:
            variances = df.var().sort_values(ascending=False)
            selected = list(variances.index[: self.target_dim])
            return selected

        return list(df.columns)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()

        # Drop rows with any missing values (paper drops 11 rows)
        df = df.dropna()

        # Drop identifier columns
        for col in self.drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Ensure dtypes and numeric conversions
        df = self._ensure_dtypes(df)

        # Numeric coercion can introduce NaNs (e.g. TotalCharges blanks); remove
        # them here so the downstream feature matrix is fully finite for KMeans.
        df = df.dropna()

        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in dataframe")

        y_raw = df[self.target_column].values

        # Label encode target
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw)

        # Drop target from features
        X_df = df.drop(columns=[self.target_column])

        # One-hot encode categorical features
        ohe_arr, ohe_names = self._one_hot_encode(X_df)

        # Numeric continuous columns (ensure ordering)
        numeric_cols = [c for c in X_df.columns if c not in ohe_names and pd.api.types.is_numeric_dtype(X_df[c])]
        # Actually, after OHE we should treat original numeric columns explicitly
        numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        numeric_arr = X_df[numeric_cols].to_numpy(dtype=float) if numeric_cols else np.zeros((len(X_df), 0))

        # Concatenate features: numeric then ohe
        if numeric_arr.size and ohe_arr.size:
            full_arr = np.hstack([numeric_arr, ohe_arr])
            full_feature_names = numeric_cols + ohe_names
        elif ohe_arr.size:
            full_arr = ohe_arr
            full_feature_names = ohe_names
        else:
            full_arr = numeric_arr
            full_feature_names = numeric_cols

        # Convert to DataFrame for selection
        full_df = pd.DataFrame(full_arr, columns=full_feature_names)

        # Correlation-based feature selection to target_dim
        selected_names = self._select_by_correlation(full_df)

        # Save feature names
        self.feature_names_ = full_feature_names
        self.selected_feature_names_ = selected_names

        X_selected = full_df[selected_names].to_numpy(dtype=float)

        # Final MinMax scaling over selected features
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_selected)

        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.ohe is None or self.scaler is None or self.label_encoder is None or self.selected_feature_names_ is None:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform first.")

        df = df.copy()
        for col in self.drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = self._ensure_dtypes(df)

        # Keep transform finite as well; prefer dropping rows with unresolved NaNs
        # rather than passing imputed values into the quantum pipeline.
        df = df.dropna()

        if self.target_column in df.columns:
            X_df = df.drop(columns=[self.target_column])
        else:
            X_df = df

        # Reconstruct numeric and categorical columns to align with fit
        cat_cols = [c for c in X_df.columns if pd.api.types.is_categorical_dtype(X_df[c])]
        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]

        ohe_arr = self.ohe.transform(X_df[cat_cols]) if cat_cols else np.zeros((len(X_df), 0))
        numeric_arr = X_df[num_cols].to_numpy(dtype=float) if num_cols else np.zeros((len(X_df), 0))

        if numeric_arr.size and ohe_arr.size:
            full_arr = np.hstack([numeric_arr, ohe_arr])
            full_feature_names = num_cols + list(sum([[f"{c}={v}" for v in cats] for c, cats in zip(cat_cols, self.ohe.categories_)], []))
        elif ohe_arr.size:
            full_arr = ohe_arr
            full_feature_names = list(sum([[f"{c}={v}" for v in cats] for c, cats in zip(cat_cols, self.ohe.categories_)], []))
        else:
            full_arr = numeric_arr
            full_feature_names = num_cols

        full_df = pd.DataFrame(full_arr, columns=full_feature_names)

        # Some features selected during fit may be missing in transform; fill missing with zeros
        for feat in self.selected_feature_names_:
            if feat not in full_df.columns:
                full_df[feat] = 0.0

        X_selected = full_df[self.selected_feature_names_].to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X_selected)

        return X_scaled

    def get_feature_names(self) -> List[str]:
        return list(self.selected_feature_names_ or [])

    def save(self, path: str) -> None:
        p = Path(path)
        with p.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "TelcoChurnPreprocessor":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        with p.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, TelcoChurnPreprocessor):
            raise TypeError("Loaded object is not a TelcoChurnPreprocessor")
        return obj
