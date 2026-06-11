"""
models/dymgnn.py — Dynamic Multilayer Graph Neural Network (DYMGNN).

Implements the full DYMGNN model family from Sections 3.4–3.5:

  GNN-RNN (Section 3.4):
    Z^(t) = GNN(X^(t), A^(t))          (Eq. 13 — topological embedding)
    H^(t) = LSTM(Z^(t), H^(t-1), C^(t-1))  (Eq. 14 — for GNN-LSTM)
    H^(t) = GRU(Z^(t), H^(t-1))             (Eq. 15 — for GNN-GRU)
    Ŷ = Decoder(H^(τ))

  GNN-RNN-ATT (Section 3.5):
    (same as above, but)
    H_att = TemporalAttention([H^(1),...,H^(τ)])  (Eq. 16–18)
    Ŷ = Decoder(H_att)

Best configuration: GAT-LSTM-ATT on double-layer network.
AUC=0.812, F1=0.851 (Table 7, Section 5).

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from dymgnn.models.gcn_layer import GCNLayer
from dymgnn.models.gat_layer import GATLayer
from dymgnn.models.temporal_attention import TemporalAttention
from dymgnn.models.decoder import Decoder


class DYMGNN(nn.Module):
    """Dynamic Multilayer Graph Neural Network (DYMGNN).

    Supports all 8 configurations from the paper:
      - GNN type: GCN or GAT
      - RNN type: LSTM or GRU
      - Attention: with or without temporal attention

    Best configuration: GAT + LSTM + attention (GAT-LSTM-ATT).

    Args:
        num_features: Input node feature dimension d (16 in paper, Table 1).
        embedding_dim: GNN/RNN embedding dimension D.
            # ASSUMED: 64 — not stated in paper (IA-01).
        num_snapshots: Number of time snapshots τ (6 in paper, Section 4.2).
        num_nodes: Number of nodes nl in the graph (varies per window).
        gnn_type: 'GCN' or 'GAT' (Section 3.2).
        rnn_type: 'LSTM' or 'GRU' (Section 3.3).
        use_attention: If True, use temporal attention (GNN-RNN-ATT, Section 3.5).
        num_gat_heads: Number of GAT attention heads.
            # ASSUMED: 4 — not stated in paper (IA-02).
        decoder_hidden1: First decoder hidden size (20, from Fig. 5).
        decoder_hidden2: Second decoder hidden size (10, from Fig. 5).
        decoder_dropout: Decoder dropout rate (0.5, from Fig. 5).
    """

    def __init__(
        self,
        num_features: int,
        embedding_dim: int,           # ASSUMED: 64 — see IA-01
        num_snapshots: int,
        num_nodes: int,
        gnn_type: str = "GAT",
        rnn_type: str = "LSTM",
        use_attention: bool = True,
        num_gat_heads: int = 4,       # ASSUMED: 4 — see IA-02
        decoder_hidden1: int = 20,    # from Fig. 5
        decoder_hidden2: int = 10,    # from Fig. 5
        decoder_dropout: float = 0.5, # from Fig. 5
    ) -> None:
        super().__init__()
        self.gnn_type = gnn_type
        self.rnn_type = rnn_type
        self.use_attention = use_attention
        self.num_snapshots = num_snapshots
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # ── Topological Encoder (Section 3.2) ───────────────────────────────
        if gnn_type == "GCN":
            self.gnn = GCNLayer(num_features, embedding_dim)
        elif gnn_type == "GAT":
            self.gnn = GATLayer(num_features, embedding_dim, num_heads=num_gat_heads)
        else:
            raise ValueError(f"gnn_type must be 'GCN' or 'GAT', got '{gnn_type}'")

        # ── Temporal Encoder (Section 3.3) ──────────────────────────────────
        if rnn_type == "LSTM":
            # LSTM operates on [nl, D] embeddings per snapshot
            # Paper Eqs. 5-9: I^(t), F^(t), C^(t), O^(t), H^(t)
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                num_layers=1,  # ASSUMED: 1 layer — see IA-04
                batch_first=False,
            )
        elif rnn_type == "GRU":
            # GRU Eqs. 10-12
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                num_layers=1,  # ASSUMED: 1 layer — see IA-04
                batch_first=False,
            )
        else:
            raise ValueError(f"rnn_type must be 'LSTM' or 'GRU', got '{rnn_type}'")

        # ── Temporal Attention (Section 3.5, Eq. 16–18) — optional ─────────
        self.attention: Optional[TemporalAttention] = None
        if use_attention:
            self.attention = TemporalAttention(embedding_dim, num_nodes)

        # ── Decoder (Section 3.6, Fig. 5) ───────────────────────────────────
        self.decoder = Decoder(
            in_dim=embedding_dim,
            hidden1=decoder_hidden1,
            hidden2=decoder_hidden2,
            dropout=decoder_dropout,
        )

    def forward(
        self,
        snapshot_feats: list[Tensor],
        snapshot_adjs: list[Tensor],
        node_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Full DYMGNN forward pass over a window of τ snapshots.

        Implements Eq. 13–18.

        Args:
            snapshot_feats: List of τ node feature matrices, each [nl, d].
            snapshot_adjs:  List of τ adjacency matrices, each [nl, nl].
            node_mask: Optional boolean mask [nl] — True = node is isolated
                (randomly dropped, Section 4.2: 50% node dropout).

        Returns:
            Y_hat: Default probabilities [nl, 1].
            betas: Temporal attention weights [tau, nl] if use_attention else None.
        """
        tau = len(snapshot_feats)
        assert tau == self.num_snapshots, (
            f"Expected {self.num_snapshots} snapshots, got {tau}"
        )

        # ── Stage 1: GNN encoding per snapshot (Eq. 13) ─────────────────────
        gnn_embeddings = []
        for t in range(tau):
            x_t = snapshot_feats[t]   # [nl, d]
            a_t = snapshot_adjs[t]    # [nl, nl]

            # Apply node dropout: isolate 50% of nodes (Section 4.2)
            if node_mask is not None and self.training:
                a_t = a_t.clone()
                a_t[node_mask, :] = 0.0
                a_t[:, node_mask] = 0.0

            z_t = self.gnn(x_t, a_t)  # [nl, D]
            gnn_embeddings.append(z_t)

        # ── Stage 2: RNN encoding over time sequence ─────────────────────────
        # Stack GNN embeddings: [tau, nl, D]
        z_seq = torch.stack(gnn_embeddings, dim=0)  # [tau, nl, D]

        # LSTM/GRU expects [seq_len, batch, input_size]
        # Here: batch = 1 (the whole graph is one "sample")
        z_seq_batched = z_seq.unsqueeze(1)  # [tau, 1, nl*D] — needs reshape

        # Treat node dimension as the "batch" for RNN:
        # z_seq is [tau, nl, D] — LSTM processes all nl nodes in parallel
        # Rearrange to [tau, nl, D] → treat nl as batch
        if self.rnn_type == "LSTM":
            # Eq. 14: H^(t), C^(t) = LSTM(Z^(t), H^(t-1), C^(t-1))
            h_seq, _ = self.rnn(z_seq)   # [tau, nl, D]
        else:
            # Eq. 15: H^(t) = GRU(Z^(t), H^(t-1))
            h_seq, _ = self.rnn(z_seq)   # [tau, nl, D]

        # ── Stage 3: Temporal Attention (Eq. 16–18) — optional ───────────────
        betas = None
        if self.use_attention and self.attention is not None:
            H_att, betas = self.attention(h_seq)  # [nl, D], [tau, nl]
            node_emb = H_att                       # [nl, D]
        else:
            # Use last hidden state H^(τ) without attention
            node_emb = h_seq[-1]  # [nl, D]

        # ── Stage 4: Decode to default probability ────────────────────────────
        y_hat = self.decoder(node_emb)  # [nl, 1]
        return y_hat, betas

    def __repr__(self) -> str:
        att_str = "+ATT" if self.use_attention else ""
        return (
            f"DYMGNN({self.gnn_type}-{self.rnn_type}{att_str}, "
            f"D={self.embedding_dim}, τ={self.num_snapshots})"
        )
