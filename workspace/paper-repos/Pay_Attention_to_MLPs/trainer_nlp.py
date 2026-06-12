"""
gmlp/training/trainer_nlp.py
-----------------------------
NLP pretraining trainer for gMLP masked language modelling.

Paper Section 4 + Appendix A.2: "Pay Attention to MLPs" (arXiv:2105.08050)

Training setup (paper Table 8):
  - Dataset: C4/English (full) or C4/RealNews (ablation)
  - Batch size: 256 (full), 2048 (ablation)
  - Steps: 1M (full), 125K (ablation)
  - Max seq len: 512 (full), 128 (ablation)
  - Optimizer: AdamW (β1=0.9, β2=0.999, ε=1e-6, wd=0.01)
  - LR: 1e-4 (full), 7e-4 (ablation) with linear warmup+decay
  - Warmup: 10K steps
  - No gradient clipping (NLP)

Paper ref: Section 4, Appendix A.2
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
from .losses import MLMLoss, build_scheduler

logger = logging.getLogger(__name__)


class NLPTrainer:
    """
    Trainer for gMLP masked language modelling pretraining.

    Supports:
      - Multi-GPU via torch.nn.parallel.DistributedDataParallel
      - Mixed precision (bfloat16 / fp16) via torch.cuda.amp
      - Gradient accumulation
      - Checkpoint save/resume
      - TensorBoard logging

    Args:
        model:     gMLP model instance (set to 'mlm' task mode).
        config:    Full gMLPConfig.
        train_loader: DataLoader for C4/MLM.
        val_loader:   DataLoader for validation perplexity.
        output_dir:   Directory for checkpoints and logs.
        device:    torch device.
    """

    def __init__(
        self,
        model: gMLP,
        config: gMLPConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
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
        mc = config.model

        self.loss_fn = MLMLoss(vocab_size=mc.vocab_size)

        # AdamW optimizer — paper Table 8
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.lr,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )

        self.scheduler = build_scheduler(
            tc.lr_schedule, self.optimizer, tc.warmup_steps, tc.num_steps
        )

        # Mixed precision scaler (disabled for bfloat16 — amp handles natively)
        self.use_amp = tc.precision in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if tc.precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(tc.precision == "fp16"))

        # Gradient clipping: paper NLP = 0 (disabled)
        self.grad_clip = tc.grad_clip

        self.global_step = 0
        self.best_val_ppl = float("inf")

        # TensorBoard (optional)
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
        except ImportError:
            logger.warning("TensorBoard not available; logging to console only.")

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> None:
        """Main training loop. Runs for config.training.num_steps steps."""
        if resume_from:
            self._load_checkpoint(resume_from)

        tc = self.config.training
        self.model.train()
        self.model.set_task("mlm")

        self._print_training_summary()

        data_iter = iter(self.train_loader)
        start_time = time.time()

        while self.global_step < tc.num_steps:
            # Fetch batch (handles IterableDataset exhaustion)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            loss = self._training_step(batch)

            # Logging
            if self.global_step % tc.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = tc.log_interval / max(elapsed, 1e-8)
                logger.info(
                    f"step={self.global_step:>7d}  loss={loss:.4f}  "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                    f"steps/s={steps_per_sec:.1f}"
                )
                if self._tb_writer:
                    self._tb_writer.add_scalar("train/loss", loss, self.global_step)
                    self._tb_writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                start_time = time.time()

            # Validation
            if self.val_loader and self.global_step % tc.eval_interval == 0 and self.global_step > 0:
                val_ppl = self.evaluate()
                logger.info(f"step={self.global_step}  val_perplexity={val_ppl:.4f}")
                if self._tb_writer:
                    self._tb_writer.add_scalar("val/perplexity", val_ppl, self.global_step)
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best")
                self.model.train()

            # Checkpoint
            if self.global_step % tc.save_interval == 0 and self.global_step > 0:
                self._save_checkpoint(f"step_{self.global_step}")

        logger.info(f"Training complete. Best val perplexity: {self.best_val_ppl:.4f}")
        self._save_checkpoint("final")

    def _training_step(self, batch: dict) -> float:
        """Single gradient update step."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            output = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = output.loss

        self.scaler.scale(loss).backward()

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

    def evaluate(self) -> float:
        """
        Compute validation perplexity (exp of mean MLM cross-entropy).
        Paper reports perplexity as the primary pretraining metric (Section 4.1).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    output = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

                total_loss += output.loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    # ---------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------

    def _print_training_summary(self) -> None:
        tc = self.config.training
        mc = self.config.model
        logger.info("=" * 60)
        logger.info("gMLP NLP Pretraining")
        logger.info(f"  Model:       {self.model}")
        logger.info(f"  Parameters:  {self.model.get_num_params():,}")
        logger.info(f"  Layers:      {mc.num_layers}  d_model={mc.d_model}  d_ffn={mc.d_ffn}")
        logger.info(f"  Tiny attn:   {mc.use_tiny_attn} (aMLP)" if mc.use_tiny_attn else f"  Tiny attn:   False (gMLP)")
        logger.info(f"  Seq len:     {mc.seq_len}")
        logger.info(f"  Toeplitz W:  {mc.use_toeplitz}")
        logger.info(f"  Total steps: {tc.num_steps:,}")
        logger.info(f"  Batch size:  {tc.batch_size}")
        logger.info(f"  Peak LR:     {tc.lr}")
        logger.info(f"  Precision:   {tc.precision}")
        logger.info(f"  Device:      {self.device}")
        logger.info("=" * 60)

    def _save_checkpoint(self, tag: str) -> None:
        path = os.path.join(self.output_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_ppl": self.best_val_ppl,
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved → {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        logger.info(f"Resumed from step {self.global_step} ({path})")
