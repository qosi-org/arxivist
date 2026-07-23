"""
Top-level LATTICE model.

Wires together: per-modality adapters -> modality-aware fusion -> spatial
graph -> TransformerConv encoder -> {reconstruction decoder, cross-modal
projection heads}. Paper reference: Figure 1, Section 3.3, Algorithm 1.

IMPLEMENTATION NOTE (design choice, not a literal paper statement):
Eq. 7 refers to a "modality-specific latent branch z^(m) after graph
refinement", which could imply the encoder maintains separate per-modality
embeddings post-fusion. The paper does not describe such a branching
mechanism in the encoder (SIR architecture is a single shared
TransformerConv stack over the *fused* representation). We therefore apply
the two ModalityProjectionHeads to the same shared embedding Z, which is the
simplest reading consistent with a single-encoder architecture and keeps the
alignment loss well-defined without inventing an unstated branching module.
This is flagged here (not in the SIR, since it is a code-level design choice
rather than a paper-parsing ambiguity) so it is easy to revisit.
"""

from __future__ import annotations

import torch
from torch import nn

from lattice.models.decoder import ReconstructionDecoder
from lattice.models.graph_encoder import LatticeGraphEncoder, SpatialGraphBuilder
from lattice.models.modality_adapters import ModalityAwareFusion, ModalityInputAdapter
from lattice.models.projection_heads import ModalityProjectionHead


class LatticeModel(nn.Module):
    """Full LATTICE model (Sections 3.1-3.4).

    Args:
        modality_dims: list of B input feature dims [d_1, ..., d_B], one per
            modality block (B=5 in the paper).
        hidden_dim: shared embedding dimension (128).
        graph_num_layers: number of TransformerConv layers (3).
        graph_num_heads: number of attention heads (4).
        graph_dropout: dropout rate (0.1).
        decoder_hidden_width: decoder hidden width (256).
        proj_hidden_width: projection head hidden width (64).
        proj_output_dim: projection head output dim (64).
        aligned_modality_pair: (a, b) indices into modality_dims used for the
            cross-modal alignment loss. ASSUMED default (0, 1) per Appendix H
            (SIR ambiguities[1], confidence 0.55).
    """

    def __init__(
        self,
        modality_dims: list[int],
        hidden_dim: int = 128,
        graph_num_layers: int = 3,
        graph_num_heads: int = 4,
        graph_dropout: float = 0.1,
        decoder_hidden_width: int = 256,
        proj_hidden_width: int = 64,
        proj_output_dim: int = 64,
        aligned_modality_pair: tuple[int, int] = (0, 1),
        knn_k: int = 6,
        edge_weight_mode: str = "uniform",
        gaussian_sigma: float | None = None,
    ) -> None:
        super().__init__()
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.hidden_dim = hidden_dim
        self.total_feature_dim = sum(modality_dims)
        self.aligned_modality_pair = aligned_modality_pair

        self.adapters = nn.ModuleList(
            [ModalityInputAdapter(d, hidden_dim) for d in modality_dims]
        )
        self.fusion = ModalityAwareFusion(num_modalities=self.num_modalities)
        self.graph_builder = SpatialGraphBuilder(
            k=knn_k, edge_weight_mode=edge_weight_mode, gaussian_sigma=gaussian_sigma
        )
        self.encoder = LatticeGraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=graph_num_layers,
            num_heads=graph_num_heads,
            dropout=graph_dropout,
        )
        self.decoder = ReconstructionDecoder(
            hidden_dim=hidden_dim,
            output_dim=self.total_feature_dim,
            decoder_hidden_width=decoder_hidden_width,
        )
        self.proj_head_a = ModalityProjectionHead(hidden_dim, proj_hidden_width, proj_output_dim)
        self.proj_head_b = ModalityProjectionHead(hidden_dim, proj_hidden_width, proj_output_dim)

    def _fuse(
        self, modality_blocks: list[torch.Tensor], presence_mask: torch.Tensor
    ) -> torch.Tensor:
        assert len(modality_blocks) == self.num_modalities, (
            f"Expected {self.num_modalities} modality blocks, got {len(modality_blocks)}"
        )
        adapted = [adapter(block) for adapter, block in zip(self.adapters, modality_blocks)]
        return self.fusion(adapted, presence_mask)

    def embed(
        self,
        modality_blocks: list[torch.Tensor],
        presence_mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-mode embedding: no masking, full modality blocks in.

        Args:
            modality_blocks: list of B tensors [N, d_b].
            presence_mask: [N, B] binary presence indicator.
            coords: [N, 2] spatial coordinates used to (re)build the graph.

        Returns:
            z: [N, hidden_dim] node embeddings.
        """
        fused = self._fuse(modality_blocks, presence_mask)
        edge_index, edge_weight = self.graph_builder.build(coords)
        edge_index = edge_index.to(fused.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(fused.device)
        return self.encoder(fused, edge_index, edge_weight)

    def forward(
        self,
        modality_blocks: list[torch.Tensor],
        presence_mask: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Training-mode forward pass (masked reconstruction + alignment + spatial).

        Args:
            modality_blocks: list of B RAW (unmasked) tensors [N, d_b]; the
                concatenation X = [X^(1), ..., X^(B)] is the reconstruction
                target (Eq. 1). Masking (Eq. 4) is applied to the
                *reconstruction target/input*, not to what is fed into the
                adapters -- see training/trainer.py for how `mask` is used
                together with this output.
            presence_mask: [N, B] binary modality-presence indicator.
            coords: [N, 2] spatial coordinates.
            mask: [N, D] binary reconstruction mask Omega (Eq. 4); returned
                unchanged so the trainer can compute the masked loss without
                re-deriving it.

        Returns:
            dict with keys:
                'z': [N, hidden_dim] node embeddings
                'x': [N, D] the (unmasked) concatenated input, reconstruction target
                'x_hat': [N, D] reconstruction
                'h_a', 'h_b': [N, proj_output_dim] projections for the aligned pair
                'mask': the Omega passed in, for convenience
        """
        x = torch.cat(modality_blocks, dim=1)  # Eq. 1: X = [X^(1), ..., X^(B)]
        fused = self._fuse(modality_blocks, presence_mask)
        edge_index, edge_weight = self.graph_builder.build(coords)
        edge_index = edge_index.to(fused.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(fused.device)

        z = self.encoder(fused, edge_index, edge_weight)  # Sec 3.3
        x_hat = self.decoder(z)  # Eq. 5
        h_a = self.proj_head_a(z)  # Eq. 7 (modality a of the aligned pair)
        h_b = self.proj_head_b(z)  # Eq. 7 (modality b of the aligned pair)

        return {
            "z": z,
            "x": x,
            "x_hat": x_hat,
            "h_a": h_a,
            "h_b": h_b,
            "mask": mask,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LatticeModel(num_modalities={self.num_modalities}, "
            f"hidden_dim={self.hidden_dim}, total_feature_dim={self.total_feature_dim})"
        )
