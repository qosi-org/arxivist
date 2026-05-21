# Domain: Biology — Parsing Hints (Stage 1 Enrichment)

Load alongside `agents/01_paper_parser.md` when the detected domain is **Biology**.

---

## Architecture extraction — Biology-specific rules

**Sequence models:** extract the exact sequence type being modelled:
- DNA (nucleotides: A, C, G, T/U), protein (amino acids: 20 standard), RNA
- Input representation: one-hot, k-mer embeddings, learned embeddings, ESM/BERT-style
- Sequence length distribution (variable vs fixed — note max length and padding strategy)

**Protein structure papers:** extract:
- Whether the model predicts structure (folding) or uses it as input
- Representation: all-atom, backbone-only, Cα-only, distance matrix, contact map
- Coordinate frame: global, local (reference frame per residue), torsion angles
- For AlphaFold-style models: multiple sequence alignment (MSA) depth used

**Genomics models:** extract:
- Whether input is sequence, expression (counts), or both
- Normalisation strategy (TPM, FPKM, log1p, scran) — critical for single-cell papers
- Batch correction method if applied
- Cell type annotation method

**Graph-based biology models:** extract:
- Node type (gene, protein, cell, atom)
- Edge type (PPI, regulatory, co-expression, covalent bond)
- Graph construction method and source database (STRING, BioGRID, etc.)

---

## Mathematical spec — Biology-specific rules

**Always extract:**
- Loss function including any class-weighting terms (biology datasets are highly imbalanced)
- Sequence-level vs residue-level vs structure-level loss
- Contact prediction loss (if applicable): cross-entropy on distance bins
- ELBO for VAE-based single-cell models with KL annealing schedule

---

## Evaluation — Biology-specific rules

**Sequence models:** extract exact benchmark (TAPE, ProteinGym, CAFA, CASP)
**Structure prediction:** RMSD, TM-score, GDT-TS, lDDT — extract all reported
**Single-cell:** ARI, NMI, silhouette score, batch correction metrics (kBET, LISI)
**Drug discovery:** AUC-ROC, AUPRC (preferred for imbalanced), Enrichment Factor
**Genomics:** AUPR, precision at k, Pearson/Spearman correlation for regression tasks

Always note: whether evaluation is on held-out proteins/genes (by sequence identity
threshold) or random split — the former is much harder and the distinction is critical.
