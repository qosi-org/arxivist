"""
PyTorch-Lightning module for DeepVol training.
Implements the training pipeline from Section 5.1.
"""
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from deepvol.models.deepvol import DeepVol
from deepvol.training.losses import get_loss_fn
from deepvol.evaluation.metrics import compute_all_metrics


class DeepVolLightning(pl.LightningModule):
    """
    LightningModule wrapping DeepVol.
    Optimizer: Adam, lr=1e-3 (Table 1).
    Loss: QLIKE (Table 1).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(dict(cfg))
        self.cfg = cfg
        self.model = DeepVol(cfg)
        self.loss_fn = get_loss_fn(cfg.training.loss_function)
        self._val_preds, self._val_targets = [], []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        sigma2_hat = self(x)
        loss = self.loss_fn(sigma2_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        sigma2_hat = self(x)
        loss = self.loss_fn(sigma2_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self._val_preds.append(sigma2_hat.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self._val_preds).numpy().flatten()
        targets = torch.cat(self._val_targets).numpy().flatten()
        metrics = compute_all_metrics(preds, targets)
        for k, v in metrics.items():
            self.log(f"val_{k}", v)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        t = self.cfg.training
        return torch.optim.Adam(
            self.parameters(),
            lr=t.learning_rate,
            betas=(t.beta1, t.beta2),
            eps=t.epsilon,
            weight_decay=t.weight_decay,
        )

    def __repr__(self):
        return f"DeepVolLightning(model={self.model})"
