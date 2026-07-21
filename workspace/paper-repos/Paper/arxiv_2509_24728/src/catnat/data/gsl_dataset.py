"""
Synthetic dataset for the Graph Structure Learning (GSL) experiment.

Generates (x, y) pairs by:
  1. Sampling a random graph A ~ P_{θ*}(A) from a multivariate Bernoulli
     with community structure (Figure 5, Appendix E.1)
  2. Running a GCN f_{ψ*}(x, A) on random node features x ~ N(0, σ²I)
  3. The output y = f_{ψ*}(x, A) is the supervision signal

Data split: 80/10/10 train/val/test (Appendix E.1, explicitly stated).

Paper: Section 5.1, Appendix E.1. arXiv: 2509.24728v2. ICML 2026.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from ..utils.config import GSLConfig, GSLDataConfig


def build_community_adjacency(n_nodes: int, n_communities: int) -> Tensor:
    """Build the base community graph structure used to define θ*_ij.

    Within each community, all edges exist (θ*_ij = θ* for intra-community edges).
    Between communities, no edges exist (θ*_ij = 0 for inter-community edges).

    Paper: Appendix E.1, Figure 5 — "4 communities" community structure.
    ASSUMED: n_nodes distributed evenly across communities.

    Args:
        n_nodes:       Total number of graph nodes.
        n_communities: Number of communities.

    Returns:
        Community mask, shape [n_nodes, n_nodes], 1 for intra-community pairs.
    """
    community_mask = torch.zeros(n_nodes, n_nodes)
    nodes_per_community = n_nodes // n_communities
    for c in range(n_communities):
        start = c * nodes_per_community
        end = start + nodes_per_community
        community_mask[start:end, start:end] = 1.0
    # Remove self-loops from the mask
    community_mask.fill_diagonal_(0.0)
    return community_mask


def build_theta_star(n_nodes: int, n_communities: int, theta_star: float) -> Tensor:
    """Build the true Bernoulli parameter matrix θ*.

    θ*_ij = theta_star  for intra-community edges (i,j in same community)
    θ*_ij = 0           for inter-community edges

    Paper: Appendix E.1, Eq. 33. Confidence: 0.99 (explicitly stated).

    Args:
        n_nodes:       Number of graph nodes.
        n_communities: Number of communities.
        theta_star:    Intra-community edge probability.

    Returns:
        Parameter matrix, shape [n_nodes, n_nodes].
    """
    community_mask = build_community_adjacency(n_nodes, n_communities)
    return community_mask * theta_star


class GSLDataGenerator:
    """Generates the synthetic GSL dataset.

    Uses a fixed GCN (with random but frozen weights ψ*) to produce y = f_{ψ*}(x, A).
    The GCN and its weights are fixed at generation time; only the latent graph A
    and input features x vary.

    Paper: Appendix E.1. Confidence: 0.88.

    Args:
        config: GSLDataConfig.
        model_config: GSLModelConfig (for GCN architecture).
    """

    def __init__(self, data_config: GSLDataConfig, model_config) -> None:
        self.data_config = data_config
        self.model_config = model_config

    def generate(
        self,
        n_samples: int,
        theta_star: float,
        seed: int = 42,
    ) -> Tuple[Tensor, Tensor]:
        """Generate (x, y) pairs.

        Args:
            n_samples:   Number of samples to generate.
            theta_star:  Intra-community Bernoulli probability.
            seed:        Random seed for reproducibility.

        Returns:
            Tuple of (X, Y) tensors, each shape [n_samples, n_nodes, d].
        """
        from ..models.gsl import GCNEncoder

        torch.manual_seed(seed)
        n = self.model_config.n_nodes
        n_comm = self.model_config.n_communities

        # Build θ* parameter matrix
        theta_mat = build_theta_star(n, n_comm, theta_star)  # [n, n]

        # Fixed GCN with random weights ψ* (not trained — used for data generation only)
        gnn_star = GCNEncoder(
            d_in=self.model_config.d_in,
            d_hidden=self.model_config.d_hidden,
            d_out=self.model_config.d_out,
            n_layers=self.model_config.n_gcn_layers,
        )
        # Fix ψ* — do not train these
        for p in gnn_star.parameters():
            p.requires_grad_(False)

        X_list, Y_list = [], []
        sigma_x = self.data_config.sigma_x

        for _ in range(n_samples):
            # Sample A ~ P_{θ*}(A): Bernoulli for each edge
            A = torch.bernoulli(theta_mat)               # [n, n]
            # Sample x ~ N(0, σ²_x I)
            x = torch.randn(1, n, self.model_config.d_in) * sigma_x  # [1, n, d_in]
            # Generate y = f_{ψ*}(x, A)
            with torch.no_grad():
                y = gnn_star(x, A.unsqueeze(0))          # [1, n, d_out]
            X_list.append(x.squeeze(0))   # [n, d_in]
            Y_list.append(y.squeeze(0))   # [n, d_out]

        X = torch.stack(X_list, dim=0)   # [n_samples, n, d_in]
        Y = torch.stack(Y_list, dim=0)   # [n_samples, n, d_out]
        return X, Y


class SyntheticGSLDataset(Dataset):
    """PyTorch Dataset wrapping synthetic GSL (x, y) pairs.

    Args:
        X: Node features, shape [N, n_nodes, d_in].
        Y: Target outputs, shape [N, n_nodes, d_out].
    """

    def __init__(self, X: Tensor, Y: Tensor) -> None:
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.Y[idx]

    def __repr__(self) -> str:
        return f"SyntheticGSLDataset(n={len(self)}, n_nodes={self.X.shape[1]})"


def get_gsl_dataloaders(
    config: GSLConfig,
    theta_star: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load or generate the GSL dataset and return train/val/test DataLoaders.

    Tries to load pre-generated data from disk first; generates if not found.

    Args:
        config:      GSLConfig.
        theta_star:  Override theta* value (uses config.data.theta_star if None).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    theta = theta_star if theta_star is not None else config.data.theta_star
    data_dir = Path(config.data.data_dir)
    cache_path = data_dir / f"gsl_theta{theta}_seed{config.data.seed}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            X, Y = pickle.load(f)
        print(f"Loaded GSL dataset from {cache_path}")
    else:
        print(f"Generating GSL dataset (theta*={theta}, n={config.data.n_samples})...")
        generator = GSLDataGenerator(config.data, config.model)
        X, Y = generator.generate(
            n_samples=config.data.n_samples,
            theta_star=theta,
            seed=config.data.seed,
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((X, Y), f)
        print(f"Saved to {cache_path}")

    full_dataset = SyntheticGSLDataset(X, Y)
    n = len(full_dataset)
    n_train = int(n * config.data.train_frac)
    n_val   = int(n * config.data.val_frac)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.data.seed),
    )

    kwargs = dict(
        batch_size=config.training.batch_size,
        num_workers=config.hardware.num_workers,
        pin_memory=config.hardware.pin_memory,
    )
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )
