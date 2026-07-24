"""
Classical benchmark suite (Sec 4.4): ridge, XGBoost, MLP, NN3 ensemble, poly2ridge.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


class RidgeBaseline:
    """Ridge regression, alpha=10 (Hoerl & Kennard 1970), Sec 4.4."""

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 10.0) -> np.ndarray:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def __repr__(self) -> str:  # noqa: D105
        return "RidgeBaseline()"


class XGBoostBaseline:
    """XGBoost: 150 trees, depth 3, learning_rate 0.05 (Chen & Guestrin 2016), Sec 4.4."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 3, learning_rate: float = 0.05) -> None:
        if xgb is None:
            raise ImportError("xgboost is required for XGBoostBaseline -- pip install xgboost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def __repr__(self) -> str:  # noqa: D105
        return f"XGBoostBaseline(n_estimators={self.n_estimators}, max_depth={self.max_depth})"


class MLPBaseline(torch.nn.Module):
    """Two-hidden-layer MLP, 64-32 (Sec 4.4)."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers += [torch.nn.Linear(prev, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NN3Baseline(torch.nn.Module):
    """Three-hidden-layer network, 32-16-8 (Gu et al. 2020 / Leippold et al. 2022 architecture), Sec 4.4.

    The paper ensembles 3 seeds -- see NN3Ensemble below for the averaging wrapper.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [32, 16, 8]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers += [torch.nn.Linear(prev, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NN3Ensemble:
    """Three-seed ensemble wrapper for NN3Baseline (Sec 4.4: 'three-seed ensemble').

    ASSUMED training recipe (SIR implementation_assumptions[4], confidence 0.5):
    Adam optimizer, lr=1e-3, early stopping on a validation split -- not stated
    in the paper. All exposed via config.classical_baselines.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, n_seeds: int = 3, lr: float = 1e-3, max_epochs: int = 200, patience: int = 20) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [32, 16, 8]
        self.n_seeds = n_seeds
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> np.ndarray:
        preds = []
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32)

        for seed in range(self.n_seeds):
            torch.manual_seed(seed)
            model = NN3Baseline(self.input_dim, self.hidden_dims)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = torch.nn.MSELoss()

            best_val_loss = float("inf")
            epochs_no_improve = 0
            for _epoch in range(self.max_epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(X_train_t)
                loss = criterion(pred, y_train_t)
                loss.backward()
                optimizer.step()

                if X_val is not None:
                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(model(X_val_t), y_val_t).item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            break

            model.eval()
            with torch.no_grad():
                preds.append(model(X_test_t).numpy())

        return np.mean(preds, axis=0)

    def __repr__(self) -> str:  # noqa: D105
        return f"NN3Ensemble(n_seeds={self.n_seeds})"


class Poly2RidgeBaseline:
    """Ridge on all pairwise products of the top-8 characteristics (Kozak et al. 2020 style), Sec 4.4."""

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 10.0) -> np.ndarray:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model = Ridge(alpha=alpha)
        model.fit(X_train_poly, y_train)
        return model.predict(X_test_poly)

    def __repr__(self) -> str:  # noqa: D105
        return "Poly2RidgeBaseline()"
