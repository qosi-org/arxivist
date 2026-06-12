"""
gmlp/training/trainer_vision.py
--------------------------------
ImageNet classification trainer for gMLP vision models.

Paper Section 3 + Appendix A.1: "Pay Attention to MLPs" (arXiv:2105.08050)

Training setup (paper Table 7):
  - Dataset:          ImageNet-1K (no extra data)
  - Epochs:           300
  - Batch size:       4096
  - Optimizer:        AdamW (β1=0.9, β2=0.999, ε=1e-6, wd=0.05)
  - Peak LR:          1e-3 with cosine decay
  - Warmup:           10K steps
  - Gradient clipping: 1.0
  - Augmentation:     AutoAugment, Mixup(α=0.8), CutMix(α=1.0), switch_prob=0.5
  - Label smoothing:  0.1
  - Stochastic depth: per-model (Ti=1.0, S=0.95, B=0.80)
  - No repeated augmentation (unlike DeiT)
  - No random erasing

Paper ref: Section 3, Appendix A.1, Table 7
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..models.gmlp import gMLP
from ..utils.config import gMLPConfig
from .losses import ClassificationLoss, build_scheduler

logger = logging.getLogger(__name__)


class VisionTrainer:
    """
    Trainer for gMLP ImageNet classification.

    Supports multi-GPU via DDP, mixed precision, stochastic depth,
    Mixup/CutMix (applied via MixupCutmixCollator in the DataLoader),
    and TensorBoard logging.

    Args:
        model:        gMLP model (vision mode, d_model/d_ffn/L from config).
        config:       Full gMLPConfig (training.* and data.* used).
        train_loader: DataLoader with MixupCutmixCollator for augmented batches.
        val_loader:   DataLoader for clean ImageNet validation.
        output_dir:   Checkpoint + log directory.
        device:       torch device.
    """

    def __init__(
        self,
        model: gMLP,
        config: gMLPConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        tc = config.training

        # AdamW — paper Table 7 (vision uses wd=0.05, unlike NLP wd=0.01)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.lr,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )

        # Total steps for scheduler: epochs × steps_per_epoch
        self.steps_per_epoch = len(train_loader)
        total_steps = tc.num_epochs * self.steps_per_epoch
        self.scheduler = build_scheduler(
            tc.lr_schedule, self.optimizer, tc.warmup_steps, total_steps
        )

        # Loss: cross-entropy with label smoothing=0.1 (paper Table 7)
        # Note: Mixup/CutMix produce soft labels handled separately in loss fn
        self.loss_fn = ClassificationLoss(label_smoothing=0.1)

        # Mixed precision
        self.use_amp = tc.precision in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if tc.precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(tc.precision == "fp16"))

        # Gradient clipping: 1.0 for vision (paper Table 7)
        self.grad_clip = tc.grad_clip

        self.global_step = 0
        self.best_top1 = 0.0

        # TensorBoard
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
        except ImportError:
            logger.warning("TensorBoard not available.")

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run for config.training.num_epochs epochs."""
        if resume_from:
            self._load_checkpoint(resume_from)

        tc = self.config.training
        self._print_summary()

        for epoch in range(tc.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for batch in self.train_loader:
                loss = self._training_step(batch)
                epoch_loss += loss
                n_batches += 1

                if self.global_step % tc.log_interval == 0:
                    logger.info(
                        f"epoch={epoch+1:>3d}  step={self.global_step:>7d}  "
                        f"loss={loss:.4f}  "
                        f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )
                    if self._tb_writer:
                        self._tb_writer.add_scalar("train/loss", loss, self.global_step)
                        self._tb_writer.add_scalar(
                            "train/lr", self.scheduler.get_last_lr()[0], self.global_step
                        )

            epoch_time = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)

            # Validate every epoch
            top1, top5 = self.evaluate()
            logger.info(
                f"[Epoch {epoch+1}/{tc.num_epochs}]  "
                f"avg_loss={avg_loss:.4f}  top1={top1:.2f}%  top5={top5:.2f}%  "
                f"time={epoch_time:.0f}s"
            )
            if self._tb_writer:
                self._tb_writer.add_scalar("val/top1", top1, epoch)
                self._tb_writer.add_scalar("val/top5", top5, epoch)

            if top1 > self.best_top1:
                self.best_top1 = top1
                self._save_checkpoint("best")

            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

        logger.info(f"Training complete. Best Top-1: {self.best_top1:.2f}%")
        self._save_checkpoint("final")

    def _training_step(self, batch) -> float:
        """Single gradient update step."""
        # MixupCutmixCollator returns dict; plain DataLoader returns (images, labels) tuple
        if isinstance(batch, dict):
            images = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
        else:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            output = self.model(pixel_values=images, labels=labels)
            loss = output.loss

        self.scaler.scale(loss).backward()

        # Gradient clipping: 1.0 for vision (paper Table 7)
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1

        return loss.item()

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    def evaluate(self) -> tuple[float, float]:
        """
        Compute ImageNet Top-1 and Top-5 accuracy on the validation set.
        Paper Table 2 reports Top-1 accuracy (std ≈ 0.1% across runs).
        """
        self.model.eval()
        correct_top1 = correct_top5 = total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    images = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    output = self.model(pixel_values=images)

                logits = output.logits                           # [B, 1000]
                _, pred_top5 = logits.topk(5, dim=-1)           # [B, 5]
                correct_top1 += (pred_top5[:, 0] == labels).sum().item()
                correct_top5 += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        top1 = 100.0 * correct_top1 / max(total, 1)
        top5 = 100.0 * correct_top5 / max(total, 1)
        return top1, top5

    # ---------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------

    def _print_summary(self) -> None:
        tc = self.config.training
        mc = self.config.model
        logger.info("=" * 60)
        logger.info("gMLP Vision Training (ImageNet-1K)")
        logger.info(f"  Model:        {self.model}")
        logger.info(f"  Parameters:   {self.model.get_num_params():,}")
        logger.info(f"  Layers:       {mc.num_layers}")
        logger.info(f"  d_model:      {mc.d_model}  d_ffn={mc.d_ffn}")
        logger.info(f"  Stoch depth:  survival_prob={mc.survival_prob}")
        logger.info(f"  Epochs:       {tc.num_epochs}")
        logger.info(f"  Batch size:   {tc.batch_size}")
        logger.info(f"  Peak LR:      {tc.lr}")
        logger.info(f"  Grad clip:    {self.grad_clip}")
        logger.info(f"  Precision:    {tc.precision}")
        logger.info(f"  Device:       {self.device}")
        logger.info("=" * 60)

    def _save_checkpoint(self, tag: str) -> None:
        path = os.path.join(self.output_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_top1": self.best_top1,
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint → {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_top1 = ckpt.get("best_top1", 0.0)
        logger.info(f"Resumed from step {self.global_step} ({path})")
