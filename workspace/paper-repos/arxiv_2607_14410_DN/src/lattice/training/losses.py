"""
Training losses for LATTICE: masked reconstruction, cross-modal alignment,
and spatial smoothness, plus the combined objective (Eq. 10).

Paper references:
  - Eq. 6 (main text): masked reconstruction as squared error.
  - Appendix H: "reconstruction error is computed... using the Huber loss
    with parameter delta=1.0" -- CONFLICTS with Eq. 6. SIR ambiguities[0]
    (confidence 0.6). Both are implemented; `recon_loss` config flag
    selects which one runs (default: "huber", per Appendix H).
  - Eq. 7-8: cross-modal NCE alignment loss.
  - Eq. 9 / Appendix I.2 Lemma I.1: spatial smoothness loss (graph Laplacian
    quadratic form).
  - Eq. 10: combined weighted objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse_reconstruction_loss(
    x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked squared-error reconstruction loss (Eq. 6, literal main-text form).

    L_rec = (1/|Omega|) * sum_{i,j} Omega_ij * (X_ij - X_hat_ij)^2

    Args:
        x: [N, D] ground-truth concatenated features.
        x_hat: [N, D] reconstructed features.
        mask: [N, D] binary mask Omega (1 = hidden/target entry).

    Returns:
        scalar loss.
    """
    _assert_shapes(x, x_hat, mask)
    num_masked = mask.sum().clamp(min=1.0)
    sq_err = (x - x_hat) ** 2
    return (mask * sq_err).sum() / num_masked


def masked_huber_reconstruction_loss(
    x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """Masked Huber-loss reconstruction (Appendix H implementation detail).

    SIR ambiguities[0] (confidence 0.6): Appendix H states the reported runs
    actually use Huber loss with delta=1.0 for this same objective, despite
    Eq. 6 in the main text showing squared error. This is the DEFAULT loss
    (config: training.recon_loss=huber) since it reflects the paper's stated
    implementation detail, but masked_mse_reconstruction_loss above is also
    provided and selectable via config.

    Args:
        x: [N, D] ground-truth concatenated features.
        x_hat: [N, D] reconstructed features.
        mask: [N, D] binary mask Omega (1 = hidden/target entry).
        delta: Huber loss transition point (1.0 in the paper).

    Returns:
        scalar loss.
    """
    _assert_shapes(x, x_hat, mask)
    num_masked = mask.sum().clamp(min=1.0)
    per_element = F.huber_loss(x_hat, x, delta=delta, reduction="none")
    return (mask * per_element).sum() / num_masked


def _assert_shapes(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> None:
    assert x.shape == x_hat.shape == mask.shape, (
        f"x, x_hat, mask must share shape [N, D]; got x={tuple(x.shape)}, "
        f"x_hat={tuple(x_hat.shape)}, mask={tuple(mask.shape)}"
    )


def nce_alignment_loss(
    h_a: torch.Tensor, h_b: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Noise-contrastive cross-modal alignment loss (Eq. 8).

    L_align = -sum_i log[ exp(sim(h_a_i, h_b_i)/tau) / sum_j exp(sim(h_a_i, h_b_j)/tau) ]

    Treats (h_a_i, h_b_i) — same spot, two modality projections — as a
    positive pair, and (h_a_i, h_b_j) for j != i as negatives, following the
    NCE-style objective of Gutmann & Hyvarinen (paper ref. [12]).

    Args:
        h_a: [N, d_c] projection of modality a (default: index 0, Visium RNA).
        h_b: [N, d_c] projection of modality b (default: index 1, spatial ATAC).
            SIR ambiguities[1] (confidence 0.55): the paper's reported runs
            align only this one pair; see lattice_model.py and config.yaml
            `aligned_modality_pair` for how to extend to multiple pairs.
        temperature: tau (0.1 in the paper).

    Returns:
        scalar loss.
    """
    assert h_a.shape == h_b.shape, f"h_a {tuple(h_a.shape)} and h_b {tuple(h_b.shape)} must match"
    h_a = F.normalize(h_a, dim=-1)
    h_b = F.normalize(h_b, dim=-1)
    logits = (h_a @ h_b.t()) / temperature  # [N, N] cosine similarity matrix
    targets = torch.arange(h_a.shape[0], device=h_a.device)
    return F.cross_entropy(logits, targets)


def spatial_smoothness_loss(
    z: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Spatial smoothness (graph Laplacian quadratic form) loss (Eq. 9 / Appendix I.2).

    L_spatial = sum_{(i,j) in E} w_ij * ||z_i - z_j||^2

    Args:
        z: [N, hidden_dim] node embeddings.
        edge_index: [2, E] long tensor.
        edge_weight: optional [E] float tensor; None => uniform weight 1.0
            for every edge (default assumption, SIR ambiguities[2]).

    Returns:
        scalar loss (sum, not mean, per Eq. 9's literal form; note this
        scales with graph size/density, which is expected given lambda3=0.1
        was tuned for the paper's fixed k=6 graph).
    """
    src, dst = edge_index[0], edge_index[1]
    diffs = z[src] - z[dst]
    sq_dists = (diffs ** 2).sum(dim=1)
    if edge_weight is None:
        edge_weight = torch.ones_like(sq_dists)
    return (edge_weight * sq_dists).sum()


def combined_loss(
    outputs: dict[str, torch.Tensor],
    lambda_rec: float = 1.0,
    lambda_align: float = 0.5,
    lambda_spatial: float = 0.1,
    recon_loss: str = "huber",
    recon_loss_delta: float = 1.0,
    alignment_temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Combine the three self-supervised losses per Eq. 10.

    L = lambda1*L_rec + lambda2*L_align + lambda3*L_spatial

    Args:
        outputs: the dict returned by LatticeModel.forward, containing
            'x', 'x_hat', 'mask', 'h_a', 'h_b', 'z', 'edge_index', 'edge_weight'.
        lambda_rec, lambda_align, lambda_spatial: loss weights (Appendix H:
            1.0, 0.5, 0.1 respectively).
        recon_loss: "huber" (default, ASSUMED per Appendix H) or "mse" (Eq. 6
            literal form). See SIR ambiguities[0].
        recon_loss_delta: Huber delta (only used if recon_loss="huber").
        alignment_temperature: tau for the NCE alignment loss.

    Returns:
        dict with 'total', 'reconstruction', 'alignment', 'spatial' scalar losses.
    """
    if recon_loss == "huber":
        l_rec = masked_huber_reconstruction_loss(
            outputs["x"], outputs["x_hat"], outputs["mask"], delta=recon_loss_delta
        )
    elif recon_loss == "mse":
        l_rec = masked_mse_reconstruction_loss(outputs["x"], outputs["x_hat"], outputs["mask"])
    else:
        raise ValueError(f"Unknown recon_loss {recon_loss!r}; expected 'huber' or 'mse'")

    l_align = nce_alignment_loss(outputs["h_a"], outputs["h_b"], temperature=alignment_temperature)
    l_spatial = spatial_smoothness_loss(outputs["z"], outputs["edge_index"], outputs["edge_weight"])

    total = lambda_rec * l_rec + lambda_align * l_align + lambda_spatial * l_spatial
    return {
        "total": total,
        "reconstruction": l_rec.detach(),
        "alignment": l_align.detach(),
        "spatial": l_spatial.detach(),
    }
