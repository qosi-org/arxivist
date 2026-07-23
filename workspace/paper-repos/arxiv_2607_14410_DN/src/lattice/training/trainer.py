"""
LATTICE trainer: implements Algorithm 1 (Training LATTICE) from the paper.

Per-epoch: sample a reconstruction mask, run the encoder/decoder, compute
the combined loss (Eq. 10), backpropagate with AdamW, and (since the paper
trains full-graph, per-sample) iterate over the dataset's samples as the
"batch" dimension. Early stopping is based on validation total loss.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from lattice.models.lattice_model import LatticeModel
from lattice.training.losses import combined_loss
from lattice.utils.config import RunSummary, set_seed


def sample_mask(shape: tuple[int, int], ratio: float, device: torch.device) -> torch.Tensor:
    """Sample a random binary reconstruction mask Omega (Eq. 4).

    Args:
        shape: (N, D).
        ratio: fraction of entries to mask (rho=0.15 in the paper).
        device: torch device to allocate the mask on.

    Returns:
        [N, D] binary float tensor (1 = masked/hidden target entry).
    """
    return (torch.rand(shape, device=device) < ratio).float()


class LatticeTrainer:
    """Trains a LatticeModel per Algorithm 1, with early stopping and checkpointing.

    Args:
        model: a constructed LatticeModel.
        config: full parsed config dict (see configs/config.yaml).
        checkpoint_dir: directory to write checkpoints to.
        log_every: log training metrics every N epochs.
    """

    def __init__(
        self,
        model: LatticeModel,
        config: dict[str, Any],
        checkpoint_dir: str | Path = "checkpoints",
        log_every: int = 1,
    ) -> None:
        self.model = model
        self.config = config
        self.train_cfg = config["training"]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every

        self.device = torch.device(self.train_cfg.get("device", "cpu"))
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            betas=(self.train_cfg.get("beta1", 0.9), self.train_cfg.get("beta2", 0.999)),
            weight_decay=self.train_cfg["weight_decay"],
        )

        set_seed(self.train_cfg.get("seed", 42))

    def _step(self, sample: dict[str, torch.Tensor], train: bool) -> dict[str, float]:
        modality_blocks = [b.to(self.device) for b in sample["modality_blocks"]]
        presence_mask = sample["presence_mask"].to(self.device)
        coords = sample["coords"].to(self.device)

        total_dim = sum(b.shape[1] for b in modality_blocks)
        n_spots = modality_blocks[0].shape[0]
        mask = sample_mask(
            (n_spots, total_dim), self.train_cfg["masking_ratio"], self.device
        )

        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            outputs = self.model(modality_blocks, presence_mask, coords, mask)
            losses = combined_loss(
                outputs,
                lambda_rec=self.train_cfg["lambda_rec"],
                lambda_align=self.train_cfg["lambda_align"],
                lambda_spatial=self.train_cfg["lambda_spatial"],
                recon_loss=self.train_cfg["recon_loss"],
                recon_loss_delta=self.train_cfg["recon_loss_delta"],
                alignment_temperature=self.train_cfg["alignment_temperature"],
            )

            if train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_cfg["gradient_clip_norm"]
                )
                self.optimizer.step()

        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    def fit(self, dataset: Dataset, num_epochs: int | None = None) -> dict[str, list]:
        """Train for up to `num_epochs`, with early stopping.

        The paper trains full-graph, per-sample, iterating samples as the
        outer loop unit (there is no traditional mini-batching within a
        sample's spot graph). Here, one "epoch" = one pass over all samples
        in `dataset`, holding out `val_fraction` of samples for validation.

        Args:
            dataset: a Dataset yielding dicts with 'modality_blocks',
                'presence_mask', 'coords' (see data/dataset.py).
            num_epochs: overrides config training.num_epochs if provided.

        Returns:
            history dict with 'train_loss', 'val_loss' per epoch.
        """
        num_epochs = num_epochs or self.train_cfg["num_epochs"]
        patience = self.train_cfg["early_stopping_patience"]

        n = len(dataset)
        n_val = max(1, int(n * self.config["data"].get("val_fraction", 0.1))) if n > 1 else 0
        val_indices = set(range(n - n_val, n)) if n_val > 0 else set()
        train_indices = [i for i in range(n) if i not in val_indices]

        num_params = sum(p.numel() for p in self.model.parameters())
        print(
            RunSummary(
                model_name="LatticeModel",
                num_params=num_params,
                num_samples=n,
                spots_per_sample=self.config["data"].get("spots_per_sample", -1),
                steps_per_epoch=len(train_indices),
                device=str(self.device),
                extra={"val_samples": len(val_indices), "recon_loss": self.train_cfg["recon_loss"]},
            )
        )

        history: dict[str, list] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            t0 = time.time()
            train_losses = [self._step(dataset[i], train=True) for i in train_indices]
            train_loss = sum(l["total"] for l in train_losses) / max(1, len(train_losses))

            if val_indices:
                val_losses = [self._step(dataset[i], train=False) for i in sorted(val_indices)]
                val_loss = sum(l["total"] for l in val_losses) / max(1, len(val_losses))
            else:
                val_loss = train_loss

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if epoch % self.log_every == 0:
                dt = time.time() - t0
                print(
                    f"[epoch {epoch:4d}] train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} ({dt:.1f}s)"
                )

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint(str(self.checkpoint_dir / "best.pt"), epoch, is_best=True)
            else:
                epochs_without_improvement += 1

            self.save_checkpoint(str(self.checkpoint_dir / "last.pt"), epoch, is_best=False)

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val improvement for {patience} epochs)"
                )
                break

        return history

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            path: destination file path.
            epoch: current epoch index.
            is_best: whether this is the best-so-far checkpoint by val loss.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "is_best": is_best,
            },
            path,
        )
