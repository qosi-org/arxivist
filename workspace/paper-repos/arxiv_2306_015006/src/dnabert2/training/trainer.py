"""Fine-tuning trainer implementing the DNABERT-2 Appendix A.3 recipe.

AdamW (lr 3e-5, wd 0.01, betas 0.9/0.98), warmup 50 steps, batch 32; validate
periodically and keep the checkpoint with the lowest validation loss (paper Sec
5.2). Metric is task-dependent (MCC or F1).
"""
from __future__ import annotations

import math
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics


class Trainer:
    """DNABERT-2 GUE fine-tuning trainer.

    Args:
        model: DNABERT2Classifier.
        train_loader / val_loader: dataloaders yielding tokenized batches.
        cfg: training config dict.
        device: torch device.
        metric: 'mcc' or 'f1'.
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
            betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.98)),
        )
        total_steps = max(1, cfg["epochs"] * len(train_loader))
        warmup = cfg.get("warmup_steps", 50)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda s: min(1.0, s / max(1, warmup))
        )
        self.use_amp = device.type == "cuda"

    def __repr__(self) -> str:  # noqa: D105
        return f"Trainer(epochs={self.cfg['epochs']}, metric={self.metric}, amp={self.use_amp})"

    def _summary(self) -> None:
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[trainer] params={n/1e6:.1f}M | train_batches={len(self.train_loader)} "
              f"| val_batches={len(self.val_loader)} | device={self.device} | amp={self.use_amp}")

    def evaluate(self, loader: DataLoader):
        self.model.eval()
        preds, labels, loss_sum, nb = [], [], 0.0, 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="eval", leave=False):
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                y = batch["labels"].to(self.device)
                logits = self.model(ids, mask)
                loss_sum += F.cross_entropy(logits, y).item()
                nb += 1
                preds.extend(logits.argmax(-1).cpu().tolist())
                labels.extend(y.cpu().tolist())
        metrics = compute_metrics(preds, labels, self.metric)
        metrics["val_loss"] = loss_sum / max(1, nb)
        return metrics

    def fit(self) -> Dict[str, float]:
        self._summary()
        best_val_loss = float("inf")
        best: Dict[str, float] = {}
        eval_every = self.cfg.get("eval_every", 200)
        step = 0
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
                self.scheduler.step()
                step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                # Validate every N steps; keep best-by-val-loss (paper Sec 5.2).
                if step % eval_every == 0:
                    val = self.evaluate(self.val_loader)
                    if val["val_loss"] < best_val_loss:
                        best_val_loss = val["val_loss"]
                        best = val
                        torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))
                    self.model.train()
            # end-of-epoch eval
            val = self.evaluate(self.val_loader)
            print(f"[epoch {epoch+1}] {val}")
            if val["val_loss"] < best_val_loss:
                best_val_loss = val["val_loss"]
                best = val
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))
        return best
