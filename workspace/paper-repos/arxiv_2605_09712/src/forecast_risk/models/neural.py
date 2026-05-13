"""
forecast_risk.models.neural
=============================
Neural network forecasters: Feed-forward NN and Hemisphere Neural Network (HNN).

Paper: Section 3 — Models
"Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

NN description (paper):
  "Feed-forward neural network with three hidden layers (400 neurons each),
  ReLU activations, and dropout regularization (rate 0.2); trained via
  Adam optimizer with early stopping."

HNN (STUB):
  The Hemisphere Neural Network (Goulet Coulombe 2025a, 2026) has specialized
  hemispheres for (long-run expectations, short-run expectations, output gap,
  commodities) sharing a common feature core. Full implementation requires
  consulting Goulet Coulombe et al. (2026), "From Reactive to Proactive
  Volatility Modeling with Hemisphere Neural Networks."
  This file provides a simplified single-output approximation only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseForecaster


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class _FFN(nn.Module):
    """Internal PyTorch module for the feed-forward neural network."""

    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralNetworkForecaster(BaseForecaster):
    """
    Feed-Forward Neural Network Forecaster.

    Paper: Section 3 — "Three hidden layers (400 neurons each), ReLU activations,
    dropout rate 0.2; trained via Adam optimizer with early stopping."

    Args:
        hidden_sizes:      List of hidden layer widths (paper: [400, 400, 400]).
        dropout:           Dropout rate (paper: 0.2).
        learning_rate:     Adam learning rate (ASSUMED: 0.001).
        max_epochs:        Max training epochs (ASSUMED: 200).
        early_stopping:    Patience in epochs (ASSUMED: 20).
        batch_size:        Mini-batch size (ASSUMED: 64).
        device:            'cpu' or 'cuda'.
        random_state:      Random seed.
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,        # Paper-specified
        learning_rate: float = 0.001,  # ASSUMED: Adam default
        max_epochs: int = 200,         # ASSUMED
        early_stopping: int = 20,      # ASSUMED
        batch_size: int = 64,          # ASSUMED
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.hidden_sizes = hidden_sizes or [400, 400, 400]  # Paper-specified
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.random_state = random_state
        self._model = None
        self._input_mean = None
        self._input_std = None

    @property
    def name(self) -> str:
        return "NN"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Standardize inputs
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._input_mean) / self._input_std

        # Train/val split: last 20% for early stopping
        n_val = max(int(0.2 * len(y)), 1)
        X_tr, X_val = X_norm[:-n_val], X_norm[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        input_dim = X_norm.shape[1]
        self._model = _FFN(input_dim, self.hidden_sizes, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_tr_t = torch.FloatTensor(X_tr).to(self.device)
        y_tr_t = torch.FloatTensor(y_tr).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = np.inf
        patience_counter = 0
        best_state = None

        self._model.train()
        for epoch in range(self.max_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self._model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

            # Validation loss for early stopping
            self._model.eval()
            with torch.no_grad():
                val_preds = self._model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
            self._model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self._input_mean) / self._input_std
        X_t = torch.FloatTensor(X_norm).to(self.device)
        with torch.no_grad():
            preds = self._model(X_t).cpu().numpy()
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# Hemisphere Neural Network (STUB)
# ─────────────────────────────────────────────────────────────────────────────

class HemisphereNeuralNetwork(BaseForecaster):
    """
    STUB: Hemisphere Neural Network (HNN).

    The full HNN architecture is described in:
      Goulet Coulombe (2025a) — "A Neural Phillips Curve and a Deep Output Gap"
      Goulet Coulombe et al. (2026) — "From Reactive to Proactive Volatility
        Modeling with Hemisphere Neural Networks"

    This stub implements a simplified single-output approximation using a
    standard feed-forward network as a placeholder. Replace with the actual
    HNN implementation from the author's codebase.

    Paper description (Sec 3 / Appendix A):
      "A constrained neural architecture with four dedicated hemispheres
      (long-run expectations, short-run expectations, output gap, commodities)
      sharing a common feature core, enabling proactive volatility forecasting."
      "Two hidden layers with 400 neurons each, ReLU activations,
      softplus output for the variance hemisphere."
      "Blocked subsampling with B=1000 bootstrap samples."
      "Volatility emphasis constraint that breaks mean/variance indeterminacy."
    """

    def __init__(
        self,
        n_hemispheres: int = 4,           # Paper-specified
        hidden_sizes: list[int] | None = None,  # Paper-specified: [400, 400]
        dropout: float = 0.0,             # ASSUMED
        learning_rate: float = 0.001,     # ASSUMED
        bootstrap_samples: int = 1000,    # Paper-specified (Appendix A)
        max_epochs: int = 200,            # ASSUMED
        early_stopping: int = 20,         # ASSUMED
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.n_hemispheres = n_hemispheres
        self.hidden_sizes = hidden_sizes or [400, 400]  # Paper-specified
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bootstrap_samples = bootstrap_samples
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state

        # STUB: uses simple NN as placeholder
        self._delegate = NeuralNetworkForecaster(
            hidden_sizes=[400, 400],
            dropout=0.0,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            device=device,
            random_state=random_state,
        )

    @property
    def name(self) -> str:
        return "HNN"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        STUB implementation — delegates to simple NN.

        TODO: Replace with actual HNN including:
          1. Common feature core (shared input representation)
          2. Four dedicated hemispheres (long-run, short-run, output gap, commodities)
          3. Volatility emphasis constraint for mean/variance indeterminacy
          4. Blocked subsampling bootstrap (B=1000)
          5. Out-of-bag reality check for variance recalibration
          6. Softplus activation for variance hemisphere
        """
        print(
            "[HNN STUB] Using simplified NN approximation. "
            "Replace with actual HNN from Goulet Coulombe (2025a, 2026)."
        )
        self._delegate.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._delegate.predict(X)

    def predict_variance(self, X: np.ndarray) -> np.ndarray:
        """
        STUB: Predict conditional variance.
        Returns ones as placeholder (actual HNN produces calibrated variance).
        """
        return np.ones(len(X))
