"""
Loss functions for DeepVol training and evaluation.
All metrics from Section 3.2. Primary training loss: QLIKE (Table 1).
"""
import torch
import torch.nn as nn


def qlike_loss(sigma2_hat: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """
    QLIKE (Quasi Log-Likelihood) — primary training loss (Table 1 / Section 3.2).
    QLIKE(sigma2, sigma2_hat) = (1/T) * sum[ log(sigma2_hat) + sigma2 / sigma2_hat ]
    Noise-robust loss function for volatility proxies (Patton 2011).
    """
    eps = 1e-8
    return torch.mean(torch.log(sigma2_hat + eps) + sigma2 / (sigma2_hat + eps))


def mae_loss(sigma2_hat: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """MAE — Section 3.2."""
    return torch.mean(torch.abs(sigma2 - sigma2_hat))


def rmse_loss(sigma2_hat: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """RMSE — Section 3.2."""
    return torch.sqrt(torch.mean((sigma2 - sigma2_hat) ** 2))


def smape_loss(sigma2_hat: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """SMAPE — Section 3.2. Scale-independent, bounded."""
    eps = 1e-8
    return torch.mean(
        torch.abs(sigma2 - sigma2_hat) / ((torch.abs(sigma2) + torch.abs(sigma2_hat)) / 2 + eps)
    )


LOSS_FN_MAP = {
    "qlike": qlike_loss,
    "mae": mae_loss,
    "rmse": rmse_loss,
    "smape": smape_loss,
}


def get_loss_fn(name: str):
    if name not in LOSS_FN_MAP:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(LOSS_FN_MAP)}")
    return LOSS_FN_MAP[name]
