"""GSR fine-tuning trainer (Sec 4.1, S1.5.2, Fig S4).

AdamW, cosine-with-warmup schedule, 10 epochs, lr = 1/3 of pre-training lr
(3e-5 for 0.1B), weight_decay 1e-1, batch size 8, grad clip 1.0. Classification
head + cross-entropy. Keeps the best checkpoint by validation metric.
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
    """DNAGPT GSR fine-tuning trainer."""

    def __init__(self, model, train_loader, val_loader, cfg, device, metric,
                 ckpt_dir: str = "checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.metric = metric
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.1),
            betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.98)),
        )
        total = max(1, cfg["epochs"]) * max(1, len(train_loader))
        warmup = int(cfg.get("warmup_epochs", 3)) * max(1, len(train_loader))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            prog = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.grad_clip = cfg.get("grad_clip", 1.0)
        self.patience = cfg.get("early_stopping_patience", 3)
        self.use_amp = device.type == "cuda"

    def __repr__(self) -> str:  # noqa: D105
        return f"Trainer(epochs={self.cfg['epochs']}, metric={self.metric})"

    def _summary(self) -> None:
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[trainer] params={n/1e6:.1f}M | train_batches={len(self.train_loader)} "
              f"| val_batches={len(self.val_loader)} | device={self.device} | amp={self.use_amp}")

    def evaluate(self, loader: DataLoader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="eval", leave=False):
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                y = batch["labels"].to(self.device)
                logits = self.model.classify(ids, mask)
                preds.extend(logits.argmax(-1).cpu().tolist())
                labels.extend(y.cpu().tolist())
        return compute_metrics(preds, labels, self.metric)

    def fit(self) -> Dict[str, float]:
        self._summary()
        best_metric = -1.0
        best: Dict[str, float] = {}
        no_improve = 0
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch+1}/{self.cfg['epochs']}")
            for batch in pbar:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                y = batch["labels"].to(self.device)
                self.optimizer.zero_grad()
                logits = self.model.classify(ids, mask)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val = self.evaluate(self.val_loader)
            score = val.get(self.metric, val["accuracy"])
            print(f"[epoch {epoch+1}] {val}")
            if score > best_metric:
                best_metric, best, no_improve = score, val, 0
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"[early-stop] no improvement for {self.patience} epochs")
                    break
        return best
