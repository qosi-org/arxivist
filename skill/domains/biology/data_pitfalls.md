# Domain: Biology — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

**Train/test leakage through sequence similarity** — the most common and most damaging
pitfall in biology ML. Sequences with high identity to training sequences appearing in
the test set inflate performance metrics dramatically. Always use MMseqs2 or CD-HIT
clustering to ensure the test set is identity-filtered.

**Database version drift** — UniProt, PDB, RefSeq, and similar databases are updated
frequently. A model trained on PDB data from 2022 evaluated on 2024 PDB data may
encounter proteins leaked from the training set. Always specify the database snapshot date.

**Label quality** — biological labels (GO terms, enzyme function, pathogenicity) are
curated and change over time. A label that was "unknown" in the training database may be
annotated in the test database, creating apparent leakage.

**Organism bias** — datasets dominated by model organisms (human, mouse, E. coli) produce
models that generalise poorly to other species. If the paper claims cross-organism
generalisation, extract the exact organism split used.

**Single-cell batch effects** — samples from different batches (sequencing runs, labs,
protocols) have systematic technical variation. Batch correction is required but not
always described in full. Flag the correction method and its parameters.

**Reference genome version** — alignment-based analyses depend critically on the reference
genome version (hg19 vs hg38 for human). Results are not comparable across genome versions.
Always extract and record the reference genome used.
