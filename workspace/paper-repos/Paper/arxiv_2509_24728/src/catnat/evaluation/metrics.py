"""
Evaluation metrics matching the paper's reported results.

Implements all metrics from Tables 2, 3, and 4:

GSL (Table 2):
  - Energy Score (ES)
  - Point Prediction MAE (PP-MAE)
  - Point Prediction MSE (PP-MSE)
  - MAE on θ (latent distribution calibration)

VAE (Table 3):
  - Test NLL via 512-sample IWAE (Burda et al. 2016)
  - ELBO

RL (Table 4):
  - Final episodic return (mean over evaluation episodes)

Paper: Sections 5.1–5.3. arXiv: 2509.24728v2. ICML 2026.
"""

from typing import Optional
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# GSL Metrics
# ---------------------------------------------------------------------------

class GSLMetrics:
    """Evaluation metrics for the Graph Structure Learning experiment (Table 2)."""

    @staticmethod
    def energy_score(preds: Tensor, targets: Tensor) -> float:
        """Compute Energy Score (Eq. 10, Appendix E.4).

        Args:
            preds:   [M, B, D] predictions
            targets: [B, D] ground truth

        Returns:
            Scalar ES value.
        """
        from ..training.losses import EnergyScore
        es_fn = EnergyScore()
        with torch.no_grad():
            return es_fn(preds, targets).item()

    @staticmethod
    def pp_mae(preds: Tensor, targets: Tensor) -> float:
        """Point Prediction MAE using the mean prediction.

        Paper: Table 2 — PP-MAE. "Lower is better."

        Args:
            preds:   [M, B, D] predictions
            targets: [B, D] ground truth

        Returns:
            Scalar MAE.
        """
        mean_pred = preds.mean(dim=0)  # [B, D]
        return (mean_pred - targets).abs().mean().item()

    @staticmethod
    def pp_mse(preds: Tensor, targets: Tensor) -> float:
        """Point Prediction MSE using the mean prediction.

        Paper: Table 2 — PP-MSE.

        Args:
            preds:   [M, B, D]
            targets: [B, D]

        Returns:
            Scalar MSE.
        """
        mean_pred = preds.mean(dim=0)
        return (mean_pred - targets).pow(2).mean().item()

    @staticmethod
    def theta_mae(theta_est: Tensor, theta_star: Tensor) -> float:
        """MAE between estimated and true latent Bernoulli parameters.

        Paper: Table 2 — "MAE on θ". "Lower is better."
        Measures calibration of the latent distribution (key metric).

        Args:
            theta_est:  Estimated edge probabilities, shape [n*n] or [n, n].
            theta_star: True edge probabilities, same shape.

        Returns:
            Scalar MAE.
        """
        return (theta_est.float() - theta_star.float()).abs().mean().item()

    @staticmethod
    def evaluate_all(model, loader: DataLoader, theta_star: Tensor, device, M: int = 32) -> dict:
        """Run all GSL metrics on a DataLoader.

        Returns:
            Dict with keys: es, pp_mae, pp_mse, theta_mae.
        """
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                B = x.shape[0]
                A_samples = model.sample_graphs(M)
                A_batch = A_samples.unsqueeze(1).expand(
                    M, B, A_samples.shape[-2], A_samples.shape[-1]
                )
                preds = model(x, A_batch)  # [M, B, N, d_out]
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        preds   = torch.cat(all_preds,   dim=1)   # [M, total_B, ...]
        targets = torch.cat(all_targets, dim=0)   # [total_B, ...]

        # Flatten spatial dims for scalar metrics
        preds_flat   = preds.reshape(M, -1, preds.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])

        # Retrieve estimated edge probabilities
        theta_est = model.get_edge_probs()[:, 1].detach().cpu()  # P(edge=1)

        return {
            "es":       GSLMetrics.energy_score(preds_flat, targets_flat),
            "pp_mae":   GSLMetrics.pp_mae(preds_flat, targets_flat),
            "pp_mse":   GSLMetrics.pp_mse(preds_flat, targets_flat),
            "theta_mae": GSLMetrics.theta_mae(theta_est, theta_star.reshape(-1)),
        }


# ---------------------------------------------------------------------------
# VAE Metrics
# ---------------------------------------------------------------------------

class VAEMetrics:
    """Evaluation metrics for the Categorical VAE experiment (Table 3)."""

    @staticmethod
    def iwae_nll(model, loader: DataLoader, device, n_samples: int = 512) -> float:
        """Compute test NLL via Importance-Weighted ELBO (Burda et al. 2016).

        Paper: Table 3 — "Negative log-likelihoods are estimated with 512 importance samples."
        Lower is better.

        Args:
            model:     CatVAE model.
            loader:    Test DataLoader.
            device:    Compute device.
            n_samples: Number of IS samples. Default 512 (explicitly stated).

        Returns:
            Scalar NLL.
        """
        model.eval()
        total_nll = 0.0
        n_batches = 0
        with torch.no_grad():
            for (x, _) in loader:
                x = x.to(device)
                nll = model.importance_weighted_nll(x, n_samples=n_samples)
                total_nll += nll.item()
                n_batches += 1
        return total_nll / n_batches

    @staticmethod
    def elbo(model, loader: DataLoader, device) -> float:
        """Compute mean ELBO on a DataLoader.

        Args:
            model:  CatVAE model.
            loader: DataLoader.
            device: Compute device.

        Returns:
            Scalar ELBO (higher is better, i.e. less negative).
        """
        model.eval()
        total = 0.0
        with torch.no_grad():
            for (x, _) in loader:
                x = x.to(device)
                out = model(x)
                total += -out["elbo_parts"]["total"].item()  # ELBO = -loss
        return total / len(loader)


# ---------------------------------------------------------------------------
# RL Metrics
# ---------------------------------------------------------------------------

class RLMetrics:
    """Evaluation metrics for the RL experiment (Table 4)."""

    @staticmethod
    def episodic_return(
        env,
        agent,
        device,
        n_episodes: int = 10,
    ) -> dict:
        """Compute mean and std episodic return over n_episodes.

        Paper: Table 4 — "Final episodic returns on Seaquest and Breakout."
        "The higher the better."

        Args:
            env:        A single preprocessed gym environment.
            agent:      PPOAgent in eval mode.
            device:     Compute device.
            n_episodes: Number of evaluation episodes.

        Returns:
            Dict with 'mean' and 'std' episodic return.
        """
        import numpy as np

        agent.eval()
        returns = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)
                with torch.no_grad():
                    out = agent.get_action_and_value(obs_t)
                action = out["action"].item()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated

            returns.append(ep_return)

        returns_arr = np.array(returns)
        return {"mean": float(returns_arr.mean()), "std": float(returns_arr.std())}
