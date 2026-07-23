"""Fine-tuning trainer implementing GENERator's Appendix C.4 recipe.

AdamW (betas 0.9/0.95, weight_decay 0.1 — same as pre-training), reduce-on-plateau
LR scheduler with early stopping (patience 5). Metric is task-dependent. Keeps the
best checkpoint by validation metric.
"""
from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics


class Trainer:
    """GENERator downstream fine-tuning trainer.

    Args:
        model: GENERatorClassifier.
        train_loader / val_loader: dataloaders yielding tokenized batches.
        cfg: training config dict.
        device: torch device.
        metric: 'accuracy' | 'mcc' | 'weighted_f1'.
        ckpt_dir: checkpoint directory.
    """

    def __init__(self, model, train_loader, val_loader, cfg, device, metric, ckpt_dir="checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.metric = metric
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        )
        # Reduce-on-plateau + early stopping (Appendix C.4).
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
        )
        self.patience = cfg.get("patience", 5)
        self.use_amp = device.type == "cuda"

    def __repr__(self) -> str:  # noqa: D105
        return f"Trainer(epochs={self.cfg['epochs']}, metric={self.metric}, amp={self.use_amp})"

    def _summary(self) -> None:
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[trainer] params={n/1e9:.2f}B | train_batches={len(self.train_loader)} "
              f"| val_batches={len(self.val_loader)} | device={self.device} | amp={self.use_amp}")

    def evaluate(self, loader: DataLoader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="eval", leave=False):
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                y = batch["labels"].to(self.device)
                logits = self.model(ids, mask)
                preds.extend(logits.argmax(-1).cpu().tolist())
                labels.extend(y.cpu().tolist())
        return compute_metrics(preds, labels, self.metric)

    def fit(self) -> Dict[str, float]:
        self._summary()
        best_metric = -1.0
        best: Dict[str, float] = {}
        primary = self.metric
        no_improve = 0
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch+1}/{self.cfg['epochs']}")
            for batch in pbar:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                y = batch["labels"].to(self.device)
                self.optimizer.zero_grad()
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = F.cross_entropy(self.model(ids, mask), y)
                else:
                    loss = F.cross_entropy(self.model(ids, mask), y)
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val = self.evaluate(self.val_loader)
            score = val.get(primary, val["accuracy"])
            self.scheduler.step(score)
            print(f"[epoch {epoch+1}] {val}")
            if score > best_metric:
                best_metric = score
                best = val
                no_improve = 0
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"[early-stop] no improvement for {self.patience} epochs")
                    break
        return best
