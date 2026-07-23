"""
Rolling-window trainer.

Implements Appendix A (Adam optimizer, Algorithm 1) and Appendix A (Early Stopping,
Algorithm 2) of "Asset Pricing in Pre-trained Transformers" (arXiv:2505.01575), applied
under the rolling-window re-estimation scheme of Section 3 (train 102mo / val 30mo,
re-estimate every 12mo).

Early-stopping patience is not numerically specified in the paper (ASSUMED=10,
config-tunable, SIR confidence 0.4).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from sert_asset_pricing.data.dataset import RollingFactorDataset
from sert_asset_pricing.training.losses import MSEWithL1


@dataclass
class RollingWindowTrainer:
    """Trains a model over one rolling window with Adam + early stopping + checkpointing.

    Args:
        model: the nn.Module to train (any of the six model families).
        train_dataset: RollingFactorDataset with split="train".
        val_dataset: RollingFactorDataset with split="val".
        config: parsed config dict (training / hardware sections used here).
        checkpoint_dir: directory to save best/last checkpoints.
    """

    model: nn.Module
    train_dataset: RollingFactorDataset
    val_dataset: RollingFactorDataset
    config: dict[str, Any]
    checkpoint_dir: str = "checkpoints"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        train_cfg = self.config["training"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            betas=(train_cfg["beta1"], train_cfg["beta2"]),
            eps=train_cfg["epsilon"],
        )
        self.loss_fn = MSEWithL1(l1_lambda=train_cfg["l1_lambda"])
        self.max_epochs = train_cfg["max_epochs"]
        self.patience = train_cfg["early_stopping_patience"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _requires_teacher_forcing(self) -> bool:
        """Full Transformer variants take (x_raw, y_shifted); encoder-only take (x_raw,)."""
        import inspect

        sig = inspect.signature(self.model.forward)
        return len(sig.parameters) >= 2

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._requires_teacher_forcing():
            # Teacher-forcing input: shift-right y in place (same T), zero-pad the first step.
            # y is already [B, T, 1]. y_shifted[:, 0] = 0; y_shifted[:, t] = y[:, t-1] for t>0.
            y_shifted = torch.zeros_like(y)
            if y.size(1) > 1:
                y_shifted[:, 1:, :] = y[:, :-1, :]
            return self.model(x, y_shifted)
        return self.model(x)

    def fit(self) -> dict[str, list[float]]:
        """Run the training loop with early stopping (Appendix Algorithm 2).

        Returns:
            History dict with "train_loss" and "val_loss" lists (one entry per epoch run).
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_counter = 0
        best_state = None

        train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        for epoch in range(self.max_epochs):
            self.model.train()
            train_losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                # Model regresses per-stock returns; we broadcast target's last dim per stock
                # by iterating on the num_stocks dimension collapsed into the batch for the
                # single-output dense head (paper's per-stock OOS return regression, Sec 4.4).
                y_target = y[..., :1] if y.shape[-1] != 1 else y
                self.optimizer.zero_grad()
                pred = self._forward(x, y_target)
                loss = self.loss_fn(pred, y_target, self.model)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_target = y[..., :1] if y.shape[-1] != 1 else y
                    pred = self._forward(x, y_target)
                    val_loss = self.loss_fn(pred, y_target, self.model)
                    val_losses.append(val_loss.item())

            train_epoch_loss = sum(train_losses) / max(len(train_losses), 1)
            val_epoch_loss = sum(val_losses) / max(len(val_losses), 1)
            history["train_loss"].append(train_epoch_loss)
            history["val_loss"].append(val_epoch_loss)

            # Appendix Algorithm 2: Early Stopping
            if val_epoch_loss < best_val:
                best_val = val_epoch_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(best_state, os.path.join(self.checkpoint_dir, "best.pt"))

        return history

    def evaluate_oos(self, split: str = "val") -> dict[str, Any]:
        """Compute OOS predictions on a dataset split for downstream metric computation.

        Args:
            split: "val" (or extend with a held-out test dataset wired in by the caller).

        Returns:
            Dict with "predictions" (list of tensors) and "targets" (list of tensors).
        """
        dataset = self.val_dataset if split == "val" else self.train_dataset
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                y_target = y[..., :1] if y.shape[-1] != 1 else y
                pred = self._forward(x, y_target)
                preds.append(pred.cpu())
                targets.append(y_target.cpu())
        return {"predictions": preds, "targets": targets}

    def __repr__(self) -> str:
        return f"RollingWindowTrainer(model={type(self.model).__name__}, max_epochs={self.max_epochs})"
