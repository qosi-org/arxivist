"""SGD trainer for CIFAR-10 ResNets — iteration-based loop matching paper Sec. 4.2.

Implements:
  - SGD with momentum 0.9, weight decay 1e-4 (excluded from BN params and biases by default).
  - StepLR drops at iterations [32000, 48000], LR /10.
  - Optional warmup at LR 0.01 for the first 400 iterations (auto-enabled for ResNet-110).
  - Best-test-accuracy checkpointing.
  - Periodic evaluation on the held-out test set (paper protocol).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet_cifar.evaluation.metrics import AccuracyMeter
from resnet_cifar.training.losses import cross_entropy_loss
from resnet_cifar.training.schedule import StepLRWithWarmup


def _split_params_for_weight_decay(
    model: nn.Module,
    weight_decay: float,
    apply_wd_to_bn: bool,
) -> list[dict]:
    """Conv/FC weights get weight_decay; BN params and biases get weight_decay=0 by default.

    This matches the modern convention; the paper does not explicitly state behavior here
    (SIR ambiguities[1], conf 0.70). Setting `apply_wd_to_bn=True` puts everything in one group.
    """
    if apply_wd_to_bn:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            for p in module.parameters(recurse=False):
                no_decay_params.append(p)
        else:
            for name, p in module.named_parameters(recurse=False):
                if name.endswith("bias"):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class Trainer:
    """End-to-end trainer for CIFAR-10 ResNets."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        val_loader: Optional[DataLoader] = None,
        output_dir: str | Path = "./runs",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg["training"]
        param_groups = _split_params_for_weight_decay(
            self.model,
            weight_decay=float(train_cfg["weight_decay"]),
            apply_wd_to_bn=bool(train_cfg.get("apply_wd_to_bn", False)),
        )
        self.optimizer = torch.optim.SGD(
            param_groups,
            lr=float(train_cfg["learning_rate"]),
            momentum=float(train_cfg["momentum"]),
            nesterov=False,
        )

        # Paper Sec. 4.2: ResNet-110 needs LR warmup at 0.01 for 400 iterations.
        # Auto-enable warmup if model.name == 'resnet110' even if config didn't say so.
        use_warmup = bool(train_cfg.get("use_warmup", False))
        if cfg["model"]["name"].lower() == "resnet110":
            use_warmup = True
        warmup_iters = int(train_cfg.get("warmup_iterations", 0)) if use_warmup else 0

        self.scheduler = StepLRWithWarmup(
            self.optimizer,
            base_lr=float(train_cfg["learning_rate"]),
            warmup_lr=float(train_cfg.get("warmup_lr", train_cfg["learning_rate"])),
            warmup_iterations=warmup_iters,
            drop_iterations=train_cfg["lr_drop_iterations"],
            drop_factor=float(train_cfg.get("lr_drop_factor", 10.0)),
        )

        self.total_iterations = int(train_cfg["total_iterations"])
        self.gradient_clipping = train_cfg.get("gradient_clipping")
        self.log_every = int(cfg["evaluation"].get("log_every_n_iterations", 100))
        self.save_best = bool(cfg["evaluation"].get("save_best_checkpoint", True))
        self.eval_every_n_epochs = int(cfg["evaluation"].get("eval_every_n_epochs", 1))

        self.history: dict[str, list] = {
            "iter": [],
            "loss": [],
            "lr": [],
            "epoch": [],
            "test_top1": [],
            "test_loss": [],
            "test_iter": [],
        }
        self.best_test_top1 = 0.0
        self.best_iter = 0

    def fit(self) -> dict:
        self._print_summary()
        iteration = 0
        epoch = 0
        loader_iter = iter(self.train_loader)
        steps_per_epoch = len(self.train_loader)
        pbar = tqdm(total=self.total_iterations, desc="train", dynamic_ncols=True)

        self.model.train()
        while iteration < self.total_iterations:
            try:
                images, labels = next(loader_iter)
            except StopIteration:
                epoch += 1
                if self.eval_every_n_epochs > 0 and epoch % self.eval_every_n_epochs == 0:
                    self._evaluate_and_record(iteration, epoch)
                    self.model.train()
                loader_iter = iter(self.train_loader)
                images, labels = next(loader_iter)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            lr = self.scheduler.step(iteration)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = cross_entropy_loss(logits, labels)
            loss.backward()
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.gradient_clipping))
            self.optimizer.step()

            if iteration % self.log_every == 0:
                self.history["iter"].append(iteration)
                self.history["loss"].append(float(loss.item()))
                self.history["lr"].append(float(lr))
                self.history["epoch"].append(epoch)
                pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.4f}", ep=epoch)

            iteration += 1
            pbar.update(1)

            if iteration % steps_per_epoch == 0:
                epoch += 1
                if self.eval_every_n_epochs > 0 and epoch % self.eval_every_n_epochs == 0:
                    self._evaluate_and_record(iteration, epoch)
                    self.model.train()

        pbar.close()

        # Final evaluation regardless of cadence.
        self._evaluate_and_record(iteration, epoch, force=True)

        if self.save_best:
            self.save_checkpoint(self.output_dir / "final.pt", extra={"history": self.history})

        return {
            "best_test_top1": self.best_test_top1,
            "best_iter": self.best_iter,
            "final_history": self.history,
        }

    def _evaluate_and_record(self, iteration: int, epoch: int, force: bool = False) -> None:
        loss, top1 = self.evaluate(self.test_loader)
        self.history["test_iter"].append(iteration)
        self.history["test_top1"].append(top1)
        self.history["test_loss"].append(loss)
        tqdm.write(
            f"[epoch {epoch} | iter {iteration}] test_loss={loss:.4f}  "
            f"test_top1={top1:.2f}%  test_err={100 - top1:.2f}%"
        )
        if self.save_best and top1 > self.best_test_top1:
            self.best_test_top1 = top1
            self.best_iter = iteration
            self.save_checkpoint(self.output_dir / "best.pt")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        meter = AccuracyMeter()
        loss_sum = 0.0
        n = 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = cross_entropy_loss(logits, labels)
            bs = labels.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs
            meter.update(logits, labels)
        avg_loss = loss_sum / max(n, 1)
        top1, _err = meter.compute()
        return avg_loss, top1

    def save_checkpoint(self, path: str | Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "best_test_top1": self.best_test_top1,
            "best_iter": self.best_iter,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return ckpt

    def _print_summary(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        steps_per_epoch = len(self.train_loader)
        epochs_total = self.total_iterations / max(steps_per_epoch, 1)
        print(f"Model: {self.cfg['model']['name']}  params={n_params:,}")
        print(f"Device: {self.device}")
        print(f"Train loader: {steps_per_epoch} steps/epoch, batch_size={self.train_loader.batch_size}")
        print(f"Total iterations: {self.total_iterations}  (~{epochs_total:.1f} epochs)")
        print(f"LR schedule: base={self.scheduler.base_lr}  drops at {self.scheduler.drop_iterations}  /{self.scheduler.drop_factor}")
        if self.scheduler.warmup_iterations > 0:
            print(f"Warmup: lr={self.scheduler.warmup_lr} for first {self.scheduler.warmup_iterations} iters")
