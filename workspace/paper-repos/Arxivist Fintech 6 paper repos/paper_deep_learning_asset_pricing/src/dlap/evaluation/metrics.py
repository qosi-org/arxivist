"""
evaluation/metrics.py — Evaluation metrics for GAN asset pricing.

Implements the three performance metrics from Section III.F:
  1. Sharpe Ratio (SR): unconditional SR of the SDF factor portfolio
  2. Explained Variation (EV): time-series R2 of individual stock returns
  3. Cross-Sectional R2 (XS-R2): cross-sectional mean R2 of pricing errors

Also implements Variable Importance (Section V.F) via average absolute gradient.

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Section III.F.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


def compute_sharpe_ratio(
    F_t: torch.Tensor,
    annualize: bool = True,
    periods_per_year: int = 12,
) -> torch.Tensor:
    """
    Compute unconditional Sharpe Ratio of the SDF factor.

    SR = E[F_t] / sqrt(Var(F_t))  (Section III.F)

    Args:
        F_t: [T] SDF factor returns (excess returns of tangency portfolio)
        annualize: multiply by sqrt(periods_per_year) for annual SR
        periods_per_year: 12 for monthly data

    Returns:
        sr: scalar Sharpe Ratio
    """
    mean = F_t.mean()
    std = F_t.std(unbiased=True)
    sr = mean / (std + 1e-8)
    if annualize:
        sr = sr * math.sqrt(periods_per_year)
    return sr


def compute_explained_variation(
    returns: torch.Tensor,
    beta: torch.Tensor,
    F_t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute time-series explained variation (EV).

    EV = 1 - [mean_t(mean_i(epsilon_{t+1,i}^2))] / [mean_t(mean_i(R^e_{t+1,i}^2))]

    where residuals epsilon_{t+1,i} are from cross-sectional projection on beta_t.

    From Section III.F: "As in Kelly et al. (2018) we do not demean returns
    due to their non-stationarity and noise in the mean estimation."

    Args:
        returns: [T, N] excess returns R^e_{t+1,i}
        beta: [T, N] risk loading estimates
        F_t: [T] SDF factor (used to compute systematic component)

    Returns:
        ev: scalar Explained Variation in [0, 1]
    """
    T, N = returns.shape

    # Compute residuals: epsilon = R^e - beta * F_t (systematic component removed)
    # Cross-sectional projection: systematic component is beta_t * F_{t+1}
    # epsilon = (I - beta(beta'beta)^{-1} beta') R^e  (Section III.F)
    # Simplified: epsilon_{t,i} = R^e_{t,i} - beta_{t,i} * F_t / (beta_t^T beta_t) * beta_{t,i}
    # For scalar factor model: residual = R^e - beta * (beta * R^e).sum() / (beta^2).sum()

    # Numerator of systematic return: beta_t^T R^e_t  [T]
    beta_sq_sum = (beta ** 2).sum(dim=-1, keepdim=True)  # [T, 1]
    beta_sq_sum = beta_sq_sum.clamp(min=1e-8)

    # Project returns onto beta
    proj_coef = (beta * returns).sum(dim=-1, keepdim=True) / beta_sq_sum  # [T, 1]
    systematic = beta * proj_coef  # [T, N]
    residuals = returns - systematic  # [T, N]  epsilon_{t,i}

    # EV = 1 - mean(epsilon^2) / mean(R^e^2)  (not demeaned per paper)
    ev = 1.0 - (residuals ** 2).mean() / ((returns ** 2).mean() + 1e-8)
    return ev


def compute_xs_r2(
    returns: torch.Tensor,
    beta: torch.Tensor,
    panel_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute cross-sectional R2 (XS-R2).

    XS-R2 = 1 - [(1/N) sum_i w_i * (mean_t[epsilon_{t,i}])^2] /
                 [(1/N) sum_i w_i * (mean_t[R^e_{t,i}])^2]

    From Section III.F: this measures how well the model explains
    the cross-section of expected returns.

    Args:
        returns: [T, N] excess returns
        beta: [T, N] risk loading estimates
        panel_weights: [N] optional T_i/T weights

    Returns:
        xs_r2: scalar cross-sectional R2
    """
    T, N = returns.shape

    # Time-average returns and residuals per stock
    beta_sq_sum = (beta ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_coef = (beta * returns).sum(dim=-1, keepdim=True) / beta_sq_sum
    systematic = beta * proj_coef
    residuals = returns - systematic  # [T, N]

    mean_resid = residuals.mean(dim=0)   # [N] E[epsilon_i]
    mean_ret = returns.mean(dim=0)       # [N] E[R^e_i]

    if panel_weights is not None:
        numerator = (panel_weights * mean_resid ** 2).mean()
        denominator = (panel_weights * mean_ret ** 2).mean()
    else:
        numerator = (mean_resid ** 2).mean()
        denominator = (mean_ret ** 2).mean()

    xs_r2 = 1.0 - numerator / (denominator + 1e-8)
    return xs_r2


def compute_variable_importance(
    model: nn.Module,
    macro_series: torch.Tensor,
    firm_chars: torch.Tensor,
    returns: torch.Tensor,
    char_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute variable importance as average absolute gradient.

    Sensitivity(x_j) = (1/C) sum_{i,t} |d omega(I_t, I_{t,i}) / d x_j|

    From Section V.F: "Our sensitivity analysis is based on the average
    absolute gradient... A sensitivity of value z for a given variable means
    that the weight w will approximately change (in magnitude) by z*Delta
    for a small change of Delta in this variable."

    Args:
        model: GANAssetPricingModel
        macro_series: [1, T, 178]
        firm_chars: [T, N, 46] — requires grad
        returns: [T, N]
        char_names: optional list of 46 characteristic names

    Returns:
        dict mapping characteristic name/index to importance score
    """
    model.eval()
    firm_chars_grad = firm_chars.clone().requires_grad_(True)

    macro_series_b = macro_series.unsqueeze(0) if macro_series.dim() == 2 else macro_series
    omega, _, _, _ = model.forward_sdf(macro_series_b, firm_chars_grad, returns)

    # Sum omega to get a scalar for backprop
    omega.abs().sum().backward()

    # Average absolute gradient over all stocks and time steps: [46]
    importance = firm_chars_grad.grad.abs().mean(dim=(0, 1))  # [46]

    # Normalize to sum to 1
    importance = importance / (importance.sum() + 1e-8)

    importance_dict = {}
    N_chars = importance.shape[0]
    for j in range(N_chars):
        name = char_names[j] if char_names else f"char_{j}"
        importance_dict[name] = importance[j].item()

    return importance_dict


def compute_all_metrics(
    returns: torch.Tensor,
    beta: torch.Tensor,
    F_t: torch.Tensor,
    panel_weights: Optional[torch.Tensor] = None,
    annualize: bool = True,
) -> Dict[str, float]:
    """
    Compute all three primary metrics at once.

    Args:
        returns: [T, N] excess returns
        beta: [T, N] risk loading estimates
        F_t: [T] SDF factor returns
        panel_weights: [N] optional panel weights
        annualize: annualize Sharpe Ratio

    Returns:
        dict with 'sharpe_ratio', 'explained_variation', 'xs_r2'
    """
    sr = compute_sharpe_ratio(F_t, annualize=annualize)
    ev = compute_explained_variation(returns, beta, F_t)
    xs_r2 = compute_xs_r2(returns, beta, panel_weights)

    return {
        "sharpe_ratio": sr.item(),
        "explained_variation": ev.item(),
        "xs_r2": xs_r2.item(),
    }
