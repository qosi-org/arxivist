"""Fine-tuning trainer for HyenaDNA downstream classification.

Implements the downstream fine-tuning loop: AdamW + cosine schedule with warmup
(SIR training_pipeline, conf 0.62 — hyperparameters are config-driven and tagged
# ASSUMED where inferred). Logs metrics, checkpoints best-by-val.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics
from ..training.losses import classification_loss


class Trainer:
    """Downstream fine-tuning trainer.

    Args:
        model: HyenaDNAClassifier.
        train_loader / val_loader: dataloaders.
        cfg: dict view of the training config section.
        device: torch device.
        metrics: list of metric names to compute.
        ckpt_dir: directory to save checkpoints.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        metrics: list,
        ckpt_dir: str = "checkpoints",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.metrics = metrics
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)),
        )
        total_steps = max(1, cfg["epochs"] * len(train_loader))
        warmup = int(cfg.get("warmup_ratio", 0.1) * total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda s: self._lr_lambda(s, warmup, total_steps)
        )
        # bf16 autocast only on CUDA; CPU falls back to fp32.
        self.use_amp = cfg.get("mixed_precision", "fp32") == "bf16" and device.type == "cuda"

    def __repr__(self) -> str:  # noqa: D105
        return f"Trainer(epochs={self.cfg['epochs']}, amp={self.use_amp})"

    @staticmethod
    def _lr_lambda(step: int, warmup: int, total: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _summary(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[trainer] params={n_params/1e6:.2f}M | train_batches={len(self.train_loader)} "
            f"| val_batches={len(self.val_loader)} | device={self.device} | amp={self.use_amp}"
        )

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc="eval", leave=False):
                x = x.to(self.device)
                logits = self.model(x)
                preds.extend(logits.argmax(-1).cpu().tolist())
                labels.extend(y.tolist())
        return compute_metrics(preds, labels, self.metrics)

    def fit(self) -> Dict[str, float]:
        self._summary()
        best_metric = -1.0
        best: Dict[str, float] = {}
        primary = self.metrics[0]
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            running = 0.0
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch+1}/{self.cfg['epochs']}")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = self.model(x)
                        loss = classification_loss(logits, y)
                else:
                    logits = self.model(x)
                    loss = classification_loss(logits, y)
                loss.backward()
                if self.cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.optimizer.step()
                self.scheduler.step()
                running += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val = self.evaluate(self.val_loader)
            print(f"[epoch {epoch+1}] train_loss={running/len(self.train_loader):.4f} val={val}")
            if val.get(primary, -1.0) > best_metric:
                best_metric = val[primary]
                best = val
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))
        return best
