"""
training/trainer.py — DYMGNN Training Loop.

Trains the model using Adam optimizer (lr=0.001) for up to 200 epochs
with early stopping (patience=50), as specified in Table C.1.

Training procedure (Section 4.2):
  - 13 rolling windows, each with 6 monthly snapshots
  - 50% node dropout per snapshot per window during training
  - BCE loss (Eq. 19) minimized with Adam

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor

from dymgnn.models.dymgnn import DYMGNN
from dymgnn.training.losses import binary_cross_entropy_loss
from dymgnn.data.dataset import FreddieDataset, FreddieWindow


class DYMGNNTrainer:
    """Full training and validation loop for DYMGNN.

    Args:
        model: DYMGNN model instance.
        cfg: Full config dictionary.
        device: torch.device.
    """

    def __init__(
        self,
        model: DYMGNN,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        train_cfg = cfg["training"]
        self.epochs: int = train_cfg["epochs"]              # 200, Table C.1
        self.patience: int = train_cfg["early_stopping_patience"]  # 50, Table C.1
        self.lr: float = train_cfg["learning_rate"]         # 0.001, Table C.1

        # Adam optimizer (Table C.1)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Node dropout probability (Section 4.2: 50%)
        self.node_dropout_p: float = cfg["network"]["node_dropout"]

        # Checkpoint settings
        log_cfg = cfg.get("logging", {})
        self.log_every: int = log_cfg.get("log_every_n_epochs", 10)
        self.ckpt_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints/"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.train_losses: list[float] = []
        self.val_aucs: list[float] = []
        self.best_val_auc: float = 0.0
        self.best_epoch: int = 0
        self.epochs_without_improvement: int = 0

    def _apply_node_dropout(self, window: FreddieWindow) -> Optional[Tensor]:
        """Randomly isolate 50% of nodes per snapshot (Section 4.2)."""
        nl = len(window)
        num_drop = int(nl * self.node_dropout_p)
        drop_indices = torch.randperm(nl)[:num_drop]
        mask = torch.zeros(nl, dtype=torch.bool, device=self.device)
        mask[drop_indices] = True
        return mask

    def _forward_window(self, window: FreddieWindow, training: bool = True) -> tuple[Tensor, Optional[Tensor]]:
        """Run model forward pass on one window."""
        feats = [f.to(self.device) for f in window.snapshot_feats]
        adjs = [a.to(self.device) for a in window.snapshot_adjs]
        node_mask = self._apply_node_dropout(window) if training else None
        return self.model(feats, adjs, node_mask)

    def train(
        self,
        train_dataset: FreddieDataset,
        val_dataset: Optional[FreddieDataset] = None,
        debug: bool = False,
    ) -> None:
        """Run full training loop.

        Args:
            train_dataset: Training windows dataset.
            val_dataset: Optional validation dataset for early stopping.
            debug: If True, run only 5 epochs for quick validation.
        """
        max_epochs = 5 if debug else self.epochs
        print(f"\n{'='*60}")
        print(f"  DYMGNN Training — {self.model}")
        print(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Train windows: {len(train_dataset)}")
        print(f"  Max epochs: {max_epochs} | Early stop patience: {self.patience}")
        print(f"  Optimizer: Adam(lr={self.lr}) | Device: {self.device}")
        print(f"{'='*60}\n")

        t_start = time.time()

        for epoch in range(1, max_epochs + 1):
            # ── Training phase ───────────────────────────────────────────────
            self.model.train()
            epoch_losses = []

            for window in train_dataset:
                self.optimizer.zero_grad()
                y_hat, _ = self._forward_window(window, training=True)
                y = window.labels.to(self.device)
                loss = binary_cross_entropy_loss(y_hat, y)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            self.train_losses.append(mean_loss)

            # ── Validation phase ─────────────────────────────────────────────
            val_auc = 0.0
            if val_dataset is not None:
                val_auc = self._evaluate_auc(val_dataset)
                self.val_aucs.append(val_auc)

                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, tag="best")
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.patience:
                    print(f"\n[Early Stopping] No improvement for {self.patience} epochs.")
                    print(f"Best AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
                    break

            # ── Logging ───────────────────────────────────────────────────────
            if epoch % self.log_every == 0 or epoch == 1:
                elapsed = time.time() - t_start
                val_str = f" | Val AUC: {val_auc:.4f}" if val_dataset else ""
                print(
                    f"Epoch {epoch:>4}/{max_epochs} | "
                    f"Loss: {mean_loss:.5f}{val_str} | "
                    f"Time: {elapsed:.0f}s"
                )

        self._save_checkpoint(epoch, tag="final")
        print(f"\nTraining complete. Best AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")

    def _evaluate_auc(self, dataset: FreddieDataset) -> float:
        """Compute AUC on a dataset (concatenating all windows)."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return 0.0

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for window in dataset:
                y_hat, _ = self._forward_window(window, training=False)
                all_preds.append(y_hat.squeeze(-1).cpu().numpy())
                all_labels.append(window.labels.numpy())

        if not all_preds:
            return 0.0

        y_score = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        if len(np.unique(y_true)) < 2:
            return 0.0  # No positive examples in window

        return float(roc_auc_score(y_true, y_score))

    def _save_checkpoint(self, epoch: int, tag: str = "") -> None:
        """Save model checkpoint."""
        fname = f"checkpoint_epoch{epoch}{'_' + tag if tag else ''}.pt"
        path = self.ckpt_dir / fname
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_auc": self.best_val_auc,
            "train_losses": self.train_losses,
        }, path)
        if tag in ("best", "final"):
            print(f"  [Checkpoint] Saved → {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint. Returns epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_auc = ckpt.get("best_val_auc", 0.0)
        self.train_losses = ckpt.get("train_losses", [])
        epoch = ckpt.get("epoch", 0)
        print(f"[Checkpoint] Loaded from {path} (epoch {epoch})")
        return epoch
