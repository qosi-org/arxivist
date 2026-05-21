# Domain: Biology — Code Conventions (Stage 4 Enrichment)

Load alongside `agents/04_code_generator.md` when domain is **Biology**.

---

## Sequence data handling

```python
from Bio import SeqIO
import torch

# Always handle variable-length sequences with padding
def collate_sequences(batch, pad_token=0):
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = True
    return padded, torch.tensor(labels), mask
```

## Sequence identity split — mandatory

**Never use random train/test split for biological sequences.**
Sequences with > threshold% identity to a training sequence must not appear in the test set.

```python
# Always implement MMseqs2-based clustering split
# or CD-HIT-based split as a fallback
def split_by_sequence_identity(sequences, threshold=0.3):
    """
    Run MMseqs2 clustering at `threshold` sequence identity.
    Assign whole clusters to train/val/test.
    This is non-negotiable for unbiased evaluation.
    """
    raise NotImplementedError(
        "Implement using MMseqs2 or CD-HIT. See data/README_data.md."
    )
```

## Single-cell data conventions

```python
import scanpy as sc
import anndata as ad

# Always use AnnData as the data container
adata = ad.AnnData(X=count_matrix, obs=cell_metadata, var=gene_metadata)

# Preprocessing pipeline (in order — order matters)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)  # TPM-style normalisation
sc.pp.log1p(adata)                              # log1p transform
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
```

## Requirements

```
torch>=2.1.0
biopython>=1.81
scanpy>=1.9.6
anndata>=0.10.0
torch-geometric>=2.4.0    # for graph models
esm>=2.0.0                # Meta ESM protein language model
pandas>=2.0.0
numpy>=1.24.0
```

## What Must NOT be done

- Do NOT use random train/test split for biological sequences
- Do NOT ignore padding masks in attention — masked positions must not contribute
- Do NOT use gene IDs from one organism as if they match another organism's IDs
- Do NOT normalise count data more than once
