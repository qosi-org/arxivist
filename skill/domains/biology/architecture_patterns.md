# Domain: Biology — Architecture Patterns (Stage 3 Enrichment)

Load alongside `agents/03_architecture_planner.md` when domain is **Biology**.

---

## Framework selection

**Default: PyTorch** for deep learning biology models.
**BioPython** for sequence parsing, alignment, and file format handling (FASTA, PDB, VCF).
**Scanpy / AnnData** for single-cell RNA-seq pipelines.
**PyG (PyTorch Geometric)** for graph neural networks on biological networks.
**ESM (Meta)** for protein language model embeddings — use pre-trained, never retrain from scratch.

## Standard biological module patterns

**Protein sequence encoder:**
```
AA_sequence → OneHot/Embedding [L, 20/D] → Conv1D stack or Transformer → Pooled [D]
```
Always use masked attention for variable-length sequences.

**Contact/distance predictor:**
```
Sequence_embedding [L, D] → Outer-product mean [L, L, D²] → ResNet2D → DistanceBins [L, L, B]
```

**scRNA-seq VAE (scVI-style):**
```
Count_matrix [N, G] → Encoder → (μ, σ) → z [N, K] → Decoder → θ [N, G] → NB likelihood
```
Negative Binomial likelihood with learnable dispersion.

**GNN for PPI/molecular:**
```
Node_features → GATConv / GCNConv (multiple layers) → Global_pooling → MLP → Label
```

## Config schema for biology papers

```yaml
model:
  sequence_length: 512      # max sequence length — pad/truncate to this
  vocab_size: 25            # 20 AA + special tokens
  embedding_dim: 256
  n_layers: 6

data:
  split_strategy: sequence_identity   # NOT random — use MMseqs2 cluster split
  identity_threshold: 0.3             # 30% sequence identity threshold
  train_organisms: null               # if organism-level split

evaluation:
  primary_metric: tm_score    # or rmsd, auprc, ari — domain-specific
```
