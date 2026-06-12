"""
gmlp/training/finetuner.py
--------------------------
NLP finetuning on GLUE (SST-2, MNLI) and SQuAD (v1.1/v2.0).

Paper Section 4.4 + Appendix A.2 (Table 9):
  "Pay Attention to MLPs" (arXiv:2105.08050)

Paper results (Table 6 — our implementation targets):
  Model       | SST-2 | MNLI-m | SQuAD1.1 | SQuAD2.0
  ------------|-------|--------|----------|----------
  gMLPbase    | 94.2  | 83.7   | 86.7     | 70.1
  aMLPbase    | 93.4  | 85.9   | 90.7     | 80.9
  gMLPlarge   | 94.8  | 86.2   | 89.5     | 78.3
  aMLPlarge   | 94.8  | 88.4   | 92.2     | 85.4

Key hyperparameters (Table 9):
  - SST-2/MNLI: batch∈{16,32}, lr∈{1e-5,2e-5,3e-5}, 5 epochs, max_seq=128
  - SQuAD:      batch=32,       lr=5e-5,              8K steps, max_seq=512
  - Evaluation: median of 5 independent runs

Paper ref: Section 4.4, Tables 6, 9
"""

from __future__ import annotations

import os
import logging
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from ..models.gmlp import gMLP
from ..utils.config import gMLPConfig
from .losses import ClassificationLoss, QALoss, build_scheduler

logger = logging.getLogger(__name__)


class NLPFinetuner:
    """
    Finetuner for NLP classification (SST-2, MNLI) and span QA (SQuAD).

    For classification: uses the CLS token representation → linear head.
    For QA: adds start/end linear heads on top of all token representations.

    Args:
        model:         Pretrained gMLP (weights loaded from MLM checkpoint).
        config:        Full gMLPConfig (finetuning hyperparameters).
        task:          One of 'sst2', 'mnli', 'squad_v1', 'squad_v2'.
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        output_dir:    Checkpoint + results directory.
        device:        torch device.
    """

    CLASSIFICATION_TASKS = ("sst2", "mnli")
    QA_TASKS = ("squad_v1", "squad_v2")
    TASK_NUM_CLASSES = {"sst2": 2, "mnli": 3}

    def __init__(
        self,
        model: gMLP,
        config: gMLPConfig,
        task: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        device: Optional[torch.device] = None,
    ) -> None:
        assert task in self.CLASSIFICATION_TASKS + self.QA_TASKS, \
            f"task must be one of {self.CLASSIFICATION_TASKS + self.QA_TASKS}"

        self.config = config
        self.task = task
        self.output_dir = output_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Switch model task mode
        if task in self.CLASSIFICATION_TASKS:
            self.model.set_task("classification")
            self.loss_fn = ClassificationLoss(label_smoothing=0.0)
        else:
            self.model.set_task("qa")
            self.loss_fn = QALoss()
            # Add QA heads: two linear layers for start/end position logits
            d_model = config.model.d_model
            self.qa_start_head = nn.Linear(d_model, 1).to(self.device)
            self.qa_end_head = nn.Linear(d_model, 1).to(self.device)

        tc = config.training
        all_params = list(self.model.parameters())
        if task in self.QA_TASKS:
            all_params += list(self.qa_start_head.parameters())
            all_params += list(self.qa_end_head.parameters())

        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=tc.lr,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )

        # Paper Table 9: 5 epochs for GLUE, 8K steps for SQuAD
        total_steps = tc.num_steps if task in self.QA_TASKS else (
            tc.num_epochs * len(train_loader) if hasattr(tc, "num_epochs") else tc.num_steps
        )
        self.total_steps = total_steps
        self.scheduler = build_scheduler(tc.lr_schedule, self.optimizer, tc.warmup_steps, total_steps)

        self.use_amp = tc.precision in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if tc.precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(tc.precision == "fp16"))

        self.global_step = 0
        self.best_metric = 0.0

    def train(self) -> float:
        """
        Run finetuning. Returns best validation metric.
        Paper evaluates with median of 5 independent runs (Table 6).
        """
        tc = self.config.training
        self.model.train()

        logger.info(f"Finetuning on {self.task.upper()}  (total steps={self.total_steps})")

        for epoch in range(getattr(tc, "num_epochs", 1)):
            for batch in self.train_loader:
                if self.global_step >= self.total_steps:
                    break
                loss = self._training_step(batch)

                if self.global_step % 100 == 0:
                    logger.info(
                        f"[{self.task}] step={self.global_step}  loss={loss:.4f}  "
                        f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )

            if self.global_step >= self.total_steps:
                break

            metric = self.evaluate()
            logger.info(f"[{self.task}] epoch={epoch+1}  val_metric={metric:.4f}")
            if metric > self.best_metric:
                self.best_metric = metric
                self._save_checkpoint("best")

        return self.best_metric

    def _training_step(self, batch: dict) -> float:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            if self.task in self.CLASSIFICATION_TASKS:
                labels = batch["labels"].to(self.device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(output.logits, labels)
            else:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = output.hidden_states                        # [B, n, d_model]
                start_logits = self.qa_start_head(hidden).squeeze(-1)  # [B, n]
                end_logits = self.qa_end_head(hidden).squeeze(-1)      # [B, n]
                loss = self.loss_fn(
                    start_logits, end_logits,
                    batch["start_positions"].to(self.device),
                    batch["end_positions"].to(self.device),
                )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        return loss.item()

    def evaluate(self) -> float:
        """Evaluate on validation set. Returns accuracy (clf) or exact-match (QA)."""
        self.model.eval()
        correct = total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                if self.task in self.CLASSIFICATION_TASKS:
                    labels = batch["labels"].to(self.device)
                    output = self.model(input_ids=input_ids)
                    preds = output.logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                else:
                    # Simplified QA accuracy (start position match)
                    output = self.model(input_ids=input_ids)
                    hidden = output.hidden_states
                    start_logits = self.qa_start_head(hidden).squeeze(-1)
                    preds = start_logits.argmax(dim=-1)
                    correct += (preds == batch["start_positions"].to(self.device)).sum().item()
                    total += preds.size(0)

        self.model.train()
        return correct / max(total, 1)

    def _save_checkpoint(self, tag: str) -> None:
        path = os.path.join(self.output_dir, f"finetune_{self.task}_{tag}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "task": self.task,
        }, path)
        logger.info(f"Finetuning checkpoint → {path}")
