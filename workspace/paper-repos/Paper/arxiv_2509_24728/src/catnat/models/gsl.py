"""
Graph Structure Learning (GSL) model.

Implements the model from Section 5.1 and Appendix E.

Architecture:
  - Latent graph distribution: Bernoulli edges parameterized by catnat (K=2)
    or sigmoid/softmax baseline (special case of catnat)
  - GNN: GCN (Kipf & Welling 2017) that maps (x, A) → y
  - Training: REINFORCE with LOO baseline on Energy Score loss

The model jointly learns:
  - θ: Bernoulli edge parameters (via REINFORCE gradient estimator)
  - ψ: GCN parameters (via direct gradient through Energy Score)

Paper: Section 5.1, Appendix E. Confidence: 0.85 (GCN dims assumed, RISK-05).
arXiv: 2509.24728v2. ICML 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..catnat import build_parameterization
from ..samplers import REINFORCESampler
from ..utils.config import GSLConfig
from ..utils.init_utils import init_scores_uniform


class GCNLayer(nn.Module):
    """Single GCN layer.

    Computes: H' = A_hat * H * W
    where A_hat = D^{-1/2} * (A + I) * D^{-1/2} is the symmetrically normalized adjacency.

    Reference: Kipf & Welling (2017). Used in Section 5.1.

    Args:
        d_in:  Input feature dimension.
        d_out: Output feature dimension.
        bias:  Whether to include a bias term. Default True.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """GCN message passing.

        Args:
            x:   Node features, shape [B, N, d_in].
            adj: Adjacency matrix (binary 0/1), shape [B, N, N] or [N, N].

        Returns:
            Updated node features, shape [B, N, d_out].
        """
        # Add self-loops: A_tilde = A + I
        N = adj.shape[-1]
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
        if adj.dim() == 2:
            A_tilde = adj + eye
        else:
            A_tilde = adj + eye.unsqueeze(0)

        # Degree matrix D^{-1/2}
        deg = A_tilde.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1] or [N, 1]
        deg_inv_sqrt = deg.pow(-0.5)

        # Symmetric normalization: A_hat = D^{-1/2} * A_tilde * D^{-1/2}
        if adj.dim() == 2:
            A_hat = deg_inv_sqrt * A_tilde * deg_inv_sqrt.transpose(-1, -2)
        else:
            A_hat = deg_inv_sqrt * A_tilde * deg_inv_sqrt.transpose(-1, -2)

        # Linear transform then propagate: H' = A_hat * (H * W)
        support = self.linear(x)              # [B, N, d_out]
        out = torch.bmm(A_hat, support) if adj.dim() == 3 else A_hat @ support
        return out

    def __repr__(self) -> str:
        return f"GCNLayer(d_in={self.linear.in_features}, d_out={self.linear.out_features})"


class GCNEncoder(nn.Module):
    """Multi-layer GCN with ReLU activations.

    Used as both the data-generating GNN and the learnable predictor f_ψ
    in the GSL experiment (Section 5.1, Appendix E.2).

    # ASSUMED (RISK-05, confidence: 0.75): n_layers=2, d_hidden=64.
    # Paper states "identical architecture" without specifying dimensions.

    Args:
        d_in:     Input feature dimension.
        d_hidden: Hidden layer dimension. ASSUMED: 64.
        d_out:    Output dimension. ASSUMED: 1 (scalar per node).
        n_layers: Number of GCN layers. ASSUMED: 2.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 64,    # ASSUMED
        d_out: int = 1,         # ASSUMED
        n_layers: int = 2,      # ASSUMED
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        dims = [d_in] + [d_hidden] * (n_layers - 1) + [d_out]
        self.layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)
        ])

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """GCN forward pass over all layers.

        Args:
            x:   Node features, shape [B, N, d_in].
            adj: Adjacency matrix, shape [B, N, N].

        Returns:
            Output features, shape [B, N, d_out].
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < self.n_layers - 1:
                h = F.relu(h)
        return h

    def __repr__(self) -> str:
        return f"GCNEncoder(layers={self.n_layers})"


class GSLModel(nn.Module):
    """Full Graph Structure Learning model.

    Jointly learns:
      - θ: Bernoulli parameters for each potential edge (catnat with K=2)
      - ψ: GCN parameters for the downstream prediction task

    Training (Appendix E.3):
      1. Sample M adjacency matrices from P_θ(A)
      2. Run GNN on each: y_pred_m = f_ψ(x, A_m)
      3. Compute Energy Score loss for each sample
      4. REINFORCE gradient for θ with LOO baseline
      5. Direct gradient for ψ

    Paper: Section 5.1, Appendix E. Confidence: 0.87.

    Args:
        config: GSLConfig dataclass.
    """

    def __init__(self, config: GSLConfig) -> None:
        super().__init__()
        self.config = config
        n = config.model.n_nodes
        self.n_nodes = n

        # Latent edge parameters: one score per potential edge (n*(n-1)/2 directed or n² full)
        # Using n² directed edges (including self-loops set to 0) for simplicity
        # ASSUMED: using n*n full adjacency (Manenti et al. 2025 convention)
        n_edges = n * n

        # Score parameters for latent Bernoulli distribution (Eq. 33, Appendix E.1)
        # Initialized from U(0, 0.1) — Appendix E.5
        init_scores = init_scores_uniform((n_edges,), low=0.0, high=0.1)
        self.score_params = nn.Parameter(init_scores)  # [n*n]

        # Parameterization π (catnat K=2 = Bernoulli, or softmax/sparsemax)
        is_catnat = config.parameterization in ("natural", "sigmoid")
        catnat_kwargs = {
            "C": config.catnat.natural_activation_C,
            "A": config.catnat.natural_activation_A,
        } if is_catnat else {}
        self.pi = build_parameterization(
            config.parameterization, K=config.catnat.K, **catnat_kwargs
        )

        # GCN predictor f_ψ
        self.gnn = GCNEncoder(
            d_in=config.model.d_in,
            d_hidden=config.model.d_hidden,
            d_out=config.model.d_out,
            n_layers=config.model.n_gcn_layers,
        )

        # Sampler
        self.sampler = REINFORCESampler()

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def get_edge_probs(self) -> Tensor:
        """Compute Bernoulli edge probabilities from score parameters.

        Returns:
            Edge probabilities, shape [n*n, 2] for catnat K=2,
            or [n*n, K] for other parameterizations.
        """
        s = self.score_params.unsqueeze(-1)   # [n*n, 1] → need [n*n, K-1] for catnat
        if hasattr(self.pi, "log_prob"):
            # catnat K=2: scores_per_var = K-1 = 1
            return self.pi(s)                  # [n*n, 2]
        else:
            # softmax/sparsemax: scores are [n*n, K]
            s_full = self.score_params.unsqueeze(-1).expand(-1, self.config.catnat.K)
            return self.pi(s_full)             # [n*n, K]

    def sample_graphs(self, n: int) -> Tensor:
        """Sample n binary adjacency matrices from the latent graph distribution.

        Paper: Appendix E.1 — P_θ(A) = Π_{i,j} θ_ij^{A_ij} * (1-θ_ij)^{1-A_ij}

        Args:
            n: Number of adjacency matrices to sample.

        Returns:
            Binary adjacency matrices, shape [n, n_nodes, n_nodes].
        """
        probs = self.get_edge_probs()            # [n*n, 2]
        # p(edge=1) is the probability of the "left" branch (class index 1 in K=2 catnat)
        p_edge = probs[:, 1]                     # [n*n]
        # Sample Bernoulli edges
        samples = torch.bernoulli(p_edge.unsqueeze(0).expand(n, -1))  # [n, n*n]
        return samples.view(n, self.n_nodes, self.n_nodes)             # [n, N, N]

    def latent_log_prob(self, A: Tensor) -> Tensor:
        """Compute log P_θ(A) = Σ_{i,j} log Bernoulli(A_ij | θ_ij).

        Used for REINFORCE gradient estimation (Appendix E.3).

        Args:
            A: Binary adjacency matrices, shape [M, n_nodes, n_nodes].

        Returns:
            Log-probabilities, shape [M].
        """
        M = A.shape[0]
        probs = self.get_edge_probs()         # [n*n, 2]
        p_edge = probs[:, 1]                  # [n*n]: P(A_ij = 1)

        A_flat = A.view(M, -1).float()        # [M, n*n]
        eps = 1e-8
        log_p = (
            A_flat * torch.log(p_edge.clamp(min=eps)).unsqueeze(0)
            + (1 - A_flat) * torch.log((1 - p_edge).clamp(min=eps)).unsqueeze(0)
        )  # [M, n*n]
        return log_p.sum(dim=-1)              # [M]

    def forward(self, x: Tensor, A_samples: Tensor) -> Tensor:
        """Predict outputs for a batch of adjacency matrices.

        Args:
            x:         Node features, shape [B, N, d_in].
            A_samples: Adjacency matrices, shape [M, B, N, N] or [M, N, N].

        Returns:
            Predictions, shape [M, B, N, d_out].
        """
        M = A_samples.shape[0]
        B = x.shape[0]
        preds = []
        for m in range(M):
            A_m = A_samples[m]          # [B, N, N] or [N, N]
            if A_m.dim() == 2:
                A_m = A_m.unsqueeze(0).expand(B, -1, -1)
            preds.append(self.gnn(x, A_m))   # [B, N, d_out]
        return torch.stack(preds, dim=0)      # [M, B, N, d_out]

    def __repr__(self) -> str:
        return (
            f"GSLModel(n_nodes={self.n_nodes}, "
            f"parameterization='{self.config.parameterization}')"
        )
