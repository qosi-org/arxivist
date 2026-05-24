"""
models/neural_net.py — Neural network models for Gu, Kelly, Xiu (2020).

Implements NN1–NN5 architectures from Section 1.7:
  NN1: [32]
  NN2: [32, 16]
  NN3: [32, 16, 8]      ← best performer in paper
  NN4: [32, 16, 8, 4]
  NN5: [32, 16, 8, 4, 2]

Architecture details (Section 1.7):
  - Fully connected feed-forward network
  - ReLU activation (Eq. 20)
  - Batch normalization at each hidden layer (Ioffe & Szegedy 2015)
  - L1 weight penalization
  - Early stopping
  - Ensemble over multiple random seeds (count ASSUMED — see config)

Recursive formula (Section 1.7):
  x_k^(l) = ReLU(x^(l-1) @ theta_k^(l-1))
  g(z; theta) = x^(L-1) @ theta^(L-1)

Paper reference: Section 1.7, Equations (18)-(20)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from asset_pricing_ml.training.losses import HuberLoss, L2Loss


class FeedForwardNN(nn.Module):
    """Single feed-forward neural network with ReLU activations and batch norm.

    Paper Section 1.7: "traditional feed-forward networks... input layer of
    raw predictors, one or more hidden layers, and an output layer."

    Architecture for NN3 (best performer):
        z: [B, 920] → Linear(920,32) → BN → ReLU
                    → Linear(32,16)  → BN → ReLU
                    → Linear(16,8)   → BN → ReLU
                    → Linear(8,1)    → r_hat: [B,1]

    Args:
        input_dim: Number of input features (920 for full predictor set).
        hidden_layers: List of hidden layer widths, e.g. [32,16,8] for NN3.
        use_batch_norm: Whether to apply batch normalization (paper: yes).
        activation: Activation function name (paper: 'relu').
    """

    def __init__(
        self,
        input_dim: int = 920,
        hidden_layers: Optional[List[int]] = None,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32, 16, 8]  # NN3 default

        if activation != "relu":
            raise ValueError(f"Paper uses ReLU. Got: {activation}")

        # Build layer sequence
        layers = []
        prev_dim = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            if use_batch_norm:
                # Paper Section 1.7: "Batch normalization cross-sectionally
                # demeans and variance standardizes the batch inputs"
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            prev_dim = width

        # Output layer: linear (no activation)
        # Paper: "g(z; theta) = x^(L-1) @ theta^(L-1)"
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.use_batch_norm = use_batch_norm

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Paper Equations (18)-(19):
          x_k^(l) = ReLU(x^(l-1) @ theta_k^(l-1))
          g(z; theta) = x^(L-1) @ theta^(L-1)

        Args:
            z: Input features [B, input_dim]

        Returns:
            Predicted excess returns [B, 1]
        """
        assert z.dim() == 2, f"Expected [B, P], got {z.shape}"
        assert z.shape[1] == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got {z.shape[1]}"
        )
        return self.network(z)

    def __repr__(self) -> str:
        return (
            f"FeedForwardNN(input={self.input_dim}, "
            f"hidden={self.hidden_layers}, bn={self.use_batch_norm})"
        )


class NeuralNetModel:
    """Full training wrapper for NN models with ensemble, early stopping, L1 reg.

    Trains an ensemble of FeedForwardNN models with different random seeds
    and averages their predictions (Section 1.7).

    Paper training details (Section 1.7):
      - Optimizer: Adam with learning rate shrinkage (Kingma & Ba 2014)
      - Regularization: L1 penalty + early stopping + batch norm + ensemble
      - Loss: L2 or Huber (Section 1.2)

    ASSUMED hyperparameters (see config.yaml):
      - batch_size: 512 (confidence 0.45)
      - n_ensemble_seeds: 10 (confidence 0.50)
      - early_stopping_patience: 5 (confidence 0.55)
      - lr: 0.001 (confidence 0.65)

    Args:
        input_dim: Number of input features (920).
        hidden_layers: Hidden layer widths (e.g. [32,16,8] for NN3).
        use_batch_norm: Apply batch normalization.
        use_huber: Use Huber loss instead of L2.
        n_ensemble_seeds: Number of random seeds for ensemble.
            # ASSUMED: not stated in paper (confidence 0.50)
        lr: Adam learning rate.
            # ASSUMED: standard default (confidence 0.65)
        l1_lambda: L1 regularization strength.
            # ASSUMED: paper states L1 is used, value not given (confidence 0.52)
        batch_size: SGD mini-batch size.
            # ASSUMED: not stated in paper (confidence 0.45)
        early_stopping_patience: Validation loss patience.
            # ASSUMED: not stated in paper (confidence 0.55)
        max_epochs: Maximum training epochs.
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        input_dim: int = 920,
        hidden_layers: Optional[List[int]] = None,
        use_batch_norm: bool = True,
        use_huber: bool = True,
        # ASSUMED: paper states ensemble but not number of seeds (confidence 0.50)
        # TODO: verify from Internet Appendix B.3
        n_ensemble_seeds: int = 10,
        # ASSUMED: Adam lr (confidence 0.65) — TODO: verify from Internet Appendix B.3
        lr: float = 0.001,
        # ASSUMED: L1 lambda (confidence 0.52) — TODO: verify from Internet Appendix B.3
        l1_lambda: float = 0.001,
        # ASSUMED: batch size (confidence 0.45) — TODO: verify from Internet Appendix B.3
        batch_size: int = 512,
        # ASSUMED: early stopping patience (confidence 0.55)
        # TODO: verify from Internet Appendix B.3
        early_stopping_patience: int = 5,
        max_epochs: int = 100,
        device: str = "cpu",
    ):
        if hidden_layers is None:
            hidden_layers = [32, 16, 8]

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.use_batch_norm = use_batch_norm
        self.use_huber = use_huber
        self.n_ensemble_seeds = n_ensemble_seeds
        self.lr = lr
        self.l1_lambda = l1_lambda
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        self.models_: List[FeedForwardNN] = []  # Ensemble members after training

    def fit(
        self,
        Z_train: np.ndarray,  # [NT_train, 920]
        R_train: np.ndarray,  # [NT_train]
        Z_val: np.ndarray,    # [NT_val, 920]
        R_val: np.ndarray,    # [NT_val]
    ) -> "NeuralNetModel":
        """Train ensemble of NNs with early stopping.

        For each seed in range(n_ensemble_seeds):
          - Initialize a fresh FeedForwardNN
          - Train with Adam + L1 + batch norm
          - Early stop on validation loss
          - Add trained model to self.models_

        Final predictions are averaged across all ensemble members.

        Args:
            Z_train: Training feature matrix [NT_train, 920]
            R_train: Training excess returns [NT_train]
            Z_val:   Validation feature matrix [NT_val, 920]
            R_val:   Validation excess returns [NT_val]

        Returns:
            self (fitted)
        """
        self.models_ = []

        Z_tr = torch.tensor(Z_train, dtype=torch.float32, device=self.device)
        R_tr = torch.tensor(R_train, dtype=torch.float32, device=self.device)
        Z_vl = torch.tensor(Z_val,   dtype=torch.float32, device=self.device)
        R_vl = torch.tensor(R_val,   dtype=torch.float32, device=self.device)

        loss_fn = HuberLoss(xi=1.0) if self.use_huber else L2Loss()
        loss_fn = loss_fn.to(self.device)

        for seed in range(self.n_ensemble_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = FeedForwardNN(
                input_dim=self.input_dim,
                hidden_layers=self.hidden_layers,
                use_batch_norm=self.use_batch_norm,
            ).to(self.device)

            # Adam optimizer — Paper: "stochastic gradient descent" with Adam
            # (Kingma & Ba 2014, Algorithm 5 in Internet Appendix B.3)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            n_train = Z_tr.shape[0]
            steps_per_epoch = max(1, n_train // self.batch_size)

            for epoch in range(self.max_epochs):
                model.train()
                # Shuffle training data
                perm = torch.randperm(n_train, device=self.device)
                epoch_loss = 0.0
                for step in range(steps_per_epoch):
                    idx = perm[step * self.batch_size : (step + 1) * self.batch_size]
                    z_batch = Z_tr[idx]
                    r_batch = R_tr[idx]

                    optimizer.zero_grad()
                    r_hat = model(z_batch).squeeze(-1)
                    loss = loss_fn(r_hat, r_batch)

                    # L1 regularization on all weight parameters
                    # Paper Section 1.7: "l1 penalization of the weight parameters"
                    l1_reg = sum(p.abs().sum() for p in model.parameters()
                                 if p.dim() > 1)
                    total_loss = loss + self.l1_lambda * l1_reg

                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Validation check for early stopping
                # Paper Section 1.7: "terminated when validation sample errors begin to increase"
                model.eval()
                with torch.no_grad():
                    val_pred = model(Z_vl).squeeze(-1)
                    val_loss = loss_fn(val_pred, R_vl).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            self.models_.append(model)

        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict excess returns by averaging ensemble members.

        Paper Section 1.7: "construct predictions by averaging forecasts
        from all networks" (ensemble reduces prediction variance).

        Args:
            Z: Feature matrix [N, 920]

        Returns:
            Predicted excess returns [N] averaged across ensemble.
        """
        if not self.models_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Z_t = torch.tensor(Z, dtype=torch.float32, device=self.device)
        preds = []

        for model in self.models_:
            model.eval()
            with torch.no_grad():
                pred = model(Z_t).squeeze(-1).cpu().numpy()
            preds.append(pred)

        # Average across ensemble members
        return np.mean(preds, axis=0)

    def get_gradients(self, Z: np.ndarray) -> np.ndarray:
        """Compute partial derivatives for SSD variable importance.

        Paper Section 1.9 (SSD measure):
            SSD_j = sum_{i,t in T1} (∂g(z; theta) / ∂z_j |_{z=z_it})^2

        Returns gradients averaged across ensemble members.

        Args:
            Z: Feature matrix [N, 920]

        Returns:
            Gradient matrix [N, 920] — ∂g/∂z_j for each sample and feature.
        """
        if not self.models_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Z_t = torch.tensor(Z, dtype=torch.float32, device=self.device, requires_grad=True)
        all_grads = []

        for model in self.models_:
            model.eval()
            pred = model(Z_t).squeeze(-1).sum()
            pred.backward()
            if Z_t.grad is not None:
                all_grads.append(Z_t.grad.cpu().numpy().copy())
                Z_t.grad.zero_()

        if not all_grads:
            return np.zeros_like(Z)

        return np.mean(all_grads, axis=0)

    def __repr__(self) -> str:
        n_fitted = len(self.models_)
        return (
            f"NeuralNetModel(hidden={self.hidden_layers}, "
            f"ensemble={n_fitted}/{self.n_ensemble_seeds}, "
            f"huber={self.use_huber})"
        )
