"""
training/trainer.py — GAN asset pricing trainer.

Implements the 3-step adversarial training procedure from Section III.D:

  Step 1 (Initialization): Update SDF network to minimize unconditional loss
      omega_hat = argmin_omega L(omega | g=constant)

  Step 2 (Adversary update): Fix SDF, maximize loss over conditional network
      g_hat = argmax_g L(omega | g)

  Step 3 (SDF update): Fix conditional, minimize conditional loss
      omega_hat = argmin_omega L(omega | g_hat)

From paper (Section III.D): "We find that our algorithm converges already after
the above three steps, i.e. the model does not improve further by repeating the
adversarial game."

Ensemble averaging (Section III.E): 9 models with different random initializations
are trained separately and their outputs averaged.

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Section III.D-E.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from dlap.models.gan_model import GANAssetPricingModel
from dlap.training.losses import (
    MomentConditionLoss,
    UnconditionalMomentLoss,
    LoadingRegressionLoss,
)
from dlap.evaluation.metrics import compute_sharpe_ratio
from dlap.utils.config import compute_panel_weights, count_parameters


class GANTrainer:
    """
    Trainer for the GAN asset pricing model.

    Orchestrates the 3-step training procedure and ensemble averaging.
    Handles the unbalanced panel structure of the CRSP dataset.

    Args:
        model: GANAssetPricingModel instance
        cfg: full configuration dict
        device: torch device
    """

    def __init__(
        self,
        model: GANAssetPricingModel,
        cfg: Dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.tcfg = cfg["training"]

        # Optimizers — separate for SDF and conditional networks
        self.sdf_optimizer = optim.Adam(
            model.sdf_parameters(),
            lr=self.tcfg["learning_rate"],
        )
        self.cond_optimizer = optim.Adam(
            model.conditional_parameters(),
            lr=self.tcfg["learning_rate"],
        )
        self.loading_optimizer = optim.Adam(
            model.loading_parameters(),
            lr=self.tcfg["learning_rate"],
        )

        # Losses
        self.moment_loss = MomentConditionLoss().to(device)
        self.unconditional_loss = UnconditionalMomentLoss().to(device)
        self.loading_loss = LoadingRegressionLoss().to(device)

        # State
        self.best_valid_sr = -float("inf")
        self.checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GANTrainer] Model parameters: {count_parameters(model):,}")
        print(f"[GANTrainer] SDF params: {sum(p.numel() for p in model.sdf_parameters()):,}")
        print(f"[GANTrainer] Conditional params: {sum(p.numel() for p in model.conditional_parameters()):,}")

    def train_epoch(
        self,
        macro_series: torch.Tensor,
        firm_chars: torch.Tensor,
        returns: torch.Tensor,
        panel_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute one full 3-step training pass.

        Args:
            macro_series: [1, T, 178] macroeconomic time series (batch=1)
            firm_chars: [T, N, 46] firm characteristics
            returns: [T, N] excess returns
            panel_weights: [N] T_i/T weights per stock

        Returns:
            dict of training metrics for this epoch
        """
        self.model.train()
        metrics = {}

        # ── Step 1: Initialize SDF with unconditional loss ─────────────────
        self.sdf_optimizer.zero_grad()
        omega, F_t, M_t, h_t = self.model.forward_sdf(macro_series, firm_chars, returns)
        loss_uncond = self.unconditional_loss(M_t, returns, panel_weights)
        loss_uncond.backward()
        self.sdf_optimizer.step()
        metrics["loss_unconditional"] = loss_uncond.item()

        # ── Step 2: Adversary maximizes pricing error ───────────────────────
        # Fix SDF, update conditional network to MAXIMIZE loss
        self.cond_optimizer.zero_grad()
        with torch.no_grad():
            # Re-compute SDF with updated weights (no gradient needed)
            _, F_t_detach, M_t_detach, _ = self.model.forward_sdf(
                macro_series, firm_chars, returns
            )
        g, _ = self.model.forward_conditional(macro_series, firm_chars)
        loss_cond = self.moment_loss(M_t_detach, returns, g, panel_weights)
        # Negate loss to maximize (adversary's objective)
        (-loss_cond).backward()
        self.cond_optimizer.step()
        metrics["loss_adversary"] = loss_cond.item()

        # ── Step 3: SDF minimizes conditional loss ──────────────────────────
        self.sdf_optimizer.zero_grad()
        omega, F_t, M_t, h_t = self.model.forward_sdf(macro_series, firm_chars, returns)
        with torch.no_grad():
            g_fixed, _ = self.model.forward_conditional(macro_series, firm_chars)
        loss_sdf = self.moment_loss(M_t, returns, g_fixed.detach(), panel_weights)
        loss_sdf.backward()
        self.sdf_optimizer.step()
        metrics["loss_sdf"] = loss_sdf.item()

        # ── Loading network (separate training) ─────────────────────────────
        self.loading_optimizer.zero_grad()
        with torch.no_grad():
            _, F_t_eval, _, h_t_eval = self.model.forward_sdf(
                macro_series, firm_chars, returns
            )
        beta_pred = self.model.forward_loadings(h_t_eval.detach(), firm_chars)
        loss_load = self.loading_loss(beta_pred, returns, F_t_eval.detach())
        loss_load.backward()
        self.loading_optimizer.step()
        metrics["loss_loading"] = loss_load.item()

        # Sharpe ratio of current SDF factor
        with torch.no_grad():
            _, F_t_sr, _, _ = self.model.forward_sdf(macro_series, firm_chars, returns)
            metrics["sharpe_ratio"] = compute_sharpe_ratio(F_t_sr, annualize=False).item()

        return metrics

    def evaluate(
        self,
        macro_series: torch.Tensor,
        firm_chars: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Run evaluation on a data split and return metrics."""
        self.model.eval()
        with torch.no_grad():
            omega, F_t, M_t, h_t = self.model.forward_sdf(
                macro_series, firm_chars, returns
            )
            sr = compute_sharpe_ratio(F_t, annualize=True).item()
        return {"sharpe_ratio": sr, "sdf_mean": F_t.mean().item(), "sdf_std": F_t.std().item()}

    def fit(
        self,
        train_data: Tuple,
        valid_data: Tuple,
        max_epochs: int = 200,
        patience: int = 20,
        log_every: int = 10,
        save_every: int = 50,
        debug: bool = False,
    ) -> Dict:
        """
        Full training loop with early stopping on validation Sharpe Ratio.

        Args:
            train_data: (macro_series, firm_chars, returns) for training split
            valid_data: same tuple for validation split
            max_epochs: maximum number of training epochs
            patience: stop if validation SR doesn't improve for this many epochs
            log_every: print metrics every N epochs
            save_every: save checkpoint every N epochs
            debug: if True, only run 2 epochs (smoke test)

        Returns:
            history: dict of metric lists over training
        """
        macro_tr, chars_tr, ret_tr = [d.to(self.device) for d in train_data]
        macro_va, chars_va, ret_va = [d.to(self.device) for d in valid_data]

        # Panel weights for training set
        T_tr = ret_tr.shape[0]
        T_i = (~ret_tr.isnan()).float().sum(dim=0)  # [N] observation counts
        panel_weights = compute_panel_weights(T_i, T_tr).to(self.device)

        history = {"train_sr": [], "valid_sr": [], "loss_sdf": []}
        no_improve = 0

        if debug:
            max_epochs = 2
            print("[DEBUG] Running 2 epochs only.")

        print(f"\n{'='*60}")
        print(f"Training: {max_epochs} epochs, patience={patience}")
        print(f"Train shape: macro={macro_tr.shape}, chars={chars_tr.shape}, ret={ret_tr.shape}")
        print(f"{'='*60}\n")

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            # Ensure batch dim for macro: [1, T, 178]
            macro_tr_b = macro_tr.unsqueeze(0) if macro_tr.dim() == 2 else macro_tr

            metrics = self.train_epoch(macro_tr_b, chars_tr, ret_tr, panel_weights)

            # Validation
            macro_va_b = macro_va.unsqueeze(0) if macro_va.dim() == 2 else macro_va
            val_metrics = self.evaluate(macro_va_b, chars_va, ret_va)

            history["train_sr"].append(metrics["sharpe_ratio"])
            history["valid_sr"].append(val_metrics["sharpe_ratio"])
            history["loss_sdf"].append(metrics["loss_sdf"])

            if epoch % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:4d} | "
                    f"L_sdf={metrics['loss_sdf']:.5f} | "
                    f"L_adv={metrics['loss_adversary']:.5f} | "
                    f"SR_train={metrics['sharpe_ratio']:.4f} | "
                    f"SR_valid={val_metrics['sharpe_ratio']:.4f} | "
                    f"t={elapsed:.2f}s"
                )

            # Checkpoint: save best model
            if val_metrics["sharpe_ratio"] > self.best_valid_sr:
                self.best_valid_sr = val_metrics["sharpe_ratio"]
                self.save_checkpoint(epoch, val_metrics["sharpe_ratio"], is_best=True)
                no_improve = 0
            else:
                no_improve += 1

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_metrics["sharpe_ratio"], is_best=False)

            # Early stopping
            if no_improve >= patience and not debug:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        print(f"\nBest validation SR: {self.best_valid_sr:.4f}")
        return history

    def save_checkpoint(self, epoch: int, valid_sr: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "sdf_optimizer_state_dict": self.sdf_optimizer.state_dict(),
            "cond_optimizer_state_dict": self.cond_optimizer.state_dict(),
            "valid_sr": valid_sr,
            "cfg": self.cfg,
        }
        if is_best:
            path = self.checkpoint_dir / self.cfg["paths"]["best_model_name"]
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint, return epoch number."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.sdf_optimizer.load_state_dict(state["sdf_optimizer_state_dict"])
        self.cond_optimizer.load_state_dict(state["cond_optimizer_state_dict"])
        print(f"Loaded checkpoint from epoch {state['epoch']} (valid SR={state['valid_sr']:.4f})")
        return state["epoch"]
