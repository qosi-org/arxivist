"""
Training loop (Sec 4.3, Fig. 6).

"The network is trained with the Adam optimiser at a learning rate of 1e-4
and weight decay of 1e-4, minimising the Huber loss ... The learning rate
is halved on plateau after 10 epochs without improvement, and training
stops early after 25 such epochs."
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spr_gnn.training.losses import HuberRegressionLoss
from spr_gnn.evaluation.metrics import RegressionMetrics


class Trainer:
    """Encapsulates the full train/validate/early-stop/checkpoint loop for arXiv:2607.18311.

    Args:
        model: a SiameseGINRegressor instance.
        config: a loaded spr_gnn.utils.config.Config.
        device: torch.device to run on.
        output_dir: directory to write checkpoints and logs to.
    """

    def __init__(self, model: torch.nn.Module, config, device: torch.device, output_dir: str) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        t = config.raw["training"]
        self.criterion = HuberRegressionLoss(delta=t.get("huber_delta", 1.0))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=t["learning_rate"],
            betas=(t.get("adam_beta1", 0.9), t.get("adam_beta2", 0.999)),
            weight_decay=t["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=t.get("lr_factor", 0.5), patience=t.get("lr_patience", 10)
        )
        self.early_stopping_patience = t.get("early_stopping_patience", 25)
        self.max_epochs = t.get("max_epochs", 200)
        self.log_every_n_steps = t.get("log_every_n_steps", 10)
        self.save_every_n_epochs = t.get("save_every_n_epochs", 5)
        self.metrics = RegressionMetrics()

    def _print_summary(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("=" * 60)
        print("SiameseGINRegressor training summary")
        print(f"  Trainable parameters : {n_params:,}")
        print(f"  Train pairs          : {len(train_loader.dataset)}")
        print(f"  Val pairs            : {len(val_loader.dataset)}")
        print(f"  Steps / epoch        : {len(train_loader)}")
        print(f"  Max epochs           : {self.max_epochs}")
        print(f"  Early stop patience  : {self.early_stopping_patience}")
        print("=" * 60)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, dict[str, float]]:
        self.model.train(mode=train)
        total_loss = 0.0
        all_preds, all_targets = [], []
        for step, (tree_a, tree_b, target) in enumerate(tqdm(loader, disable=not train, leave=False)):
            tree_a, tree_b, target = tree_a.to(self.device), tree_b.to(self.device), target.to(self.device)
            with torch.set_grad_enabled(train):
                pred = self.model(tree_a, tree_b)
                loss = self.criterion(pred, target)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            total_loss += loss.item() * target.size(0)
            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())
            if train and self.log_every_n_steps and step % self.log_every_n_steps == 0:
                print(f"    step {step}/{len(loader)} loss={loss.item():.4f}")

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, self.metrics.compute(preds, targets)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        """Runs the full training loop with early stopping and LR scheduling.

        Returns:
            History dict with per-epoch train/val loss and metrics.
        """
        self._print_summary(train_loader, val_loader)
        history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_metrics": []}
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            start = time.time()
            train_loss, _ = self._run_epoch(train_loader, train=True)
            val_loss, val_metrics = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            elapsed = time.time() - start
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_R2={val_metrics['r2']:.4f} val_MAE={val_metrics['mae']:.2f} ({elapsed:.1f}s)"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint(str(self.output_dir / "best_model.pt"))
            else:
                epochs_without_improvement += 1

            if epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(str(self.output_dir / f"epoch_{epoch:03d}.pt"))

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {self.early_stopping_patience} epochs).")
                break

        return history

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.raw,
            },
            path,
        )
