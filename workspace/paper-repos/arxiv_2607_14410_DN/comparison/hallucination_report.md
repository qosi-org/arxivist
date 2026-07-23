# Hallucination Report

**Paper**: LATTICE (arxiv_2607_14410)
**Comparison Date**: 2026-07-22
**Scope**: Compares `architecture_plan.json` / generated code against `sir.json` to identify
components that were invented (structural), incorrectly assumed (parametric), or
dropped (omission) relative to what the paper actually specifies.

---

## Structural Hallucinations

Components in the generated code that are NOT directly specified in the SIR.

### 1. Shared embedding Z routed to both cross-modal projection heads

- **Location**: `src/lattice/models/lattice_model.py::LatticeModel.forward`
- **Severity**: Minor
- **Evidence**: Eq. 7 in the paper references a "modality-specific latent
  branch $z^{(m)}$ after graph refinement" for the projection heads. This
  phrasing could imply the encoder maintains separate per-modality branches
  internally. However, neither the SIR's `architecture.modules` nor Appendix
  H describe any such branching mechanism — the encoder is a single shared
  TransformerConv stack over the fused representation (SIR confidence 0.95
  for the encoder itself). To keep the alignment loss well-defined without
  inventing an undescribed branching module, this implementation applies
  both `ModalityProjectionHead`s to the same shared `Z`. This is flagged in
  the code's own docstring as a design choice, not a literal paper claim.
- **Suggested fix**: If the authors publish reference code, check whether
  the encoder produces per-modality residual branches prior to the shared
  `Z`, and route each projection head to its own branch if so.

*No other structural hallucinations were found.* Every other module in
`architecture_plan.json → module_hierarchy` maps directly to an SIR
`architecture.modules` entry or an explicit equation.

---

## Parametric Hallucinations

Hyperparameters marked `# ASSUMED` in `configs/config.yaml` that could be
wrong, cross-referenced against where they coincide with observed deviations.

### 1. `training.recon_loss: huber` (delta=1.0)

- **Severity**: Significant
- **Evidence**: SIR `ambiguities[0]`, confidence 0.6. Main-text Eq. 6 defines
  the reconstruction loss as squared error; Appendix H states the actual
  implementation uses Huber loss. This is a genuine internal inconsistency
  in the paper, not a parsing error — both loss forms are implemented and
  selectable (`training/losses.py`).
- **Suggested fix**: Re-run with `mse` and compare reconstruction-loss
  trajectories; if the authors release code, defer to it.

### 2. `model.projection_head.aligned_modality_pair: [0, 1]`

- **Severity**: Minor
- **Evidence**: SIR `ambiguities[1]`, confidence 0.55. Eq. 7-8 are written
  generically for arbitrary modality pairs; Appendix H names only one
  concrete pair (Visium RNA, spatial ATAC). Using only this one pair may
  under-utilize the alignment objective, particularly at higher modality-
  ladder levels (M3-M5) where more blocks are active.
- **Suggested fix**: Try all-pairs alignment (`nce_alignment_loss` already
  accepts a list of pairs) and compare the M1→M5 ARI/NMI trend against
  Table 2's pattern (peak around M2-M3, decline at M4-M5).

### 3. `data.edge_weight_mode: uniform`

- **Severity**: Minor
- **Evidence**: SIR `ambiguities[2]`, confidence 0.5. Eq. 3's Gaussian kernel
  has no stated sigma value, and Appendix H doesn't confirm whether it was
  enabled for the reported spatial-smoothness loss.
- **Suggested fix**: Sweep `edge_weight_mode: gaussian` (implemented, sigma
  defaults to median pairwise distance) and compare spatial contiguity.

### 4. `training.beta1` / `training.beta2` (AdamW, PyTorch defaults 0.9/0.999)

- **Severity**: Minor
- **Evidence**: SIR `implementation_assumptions[0]`, confidence 0.6. Not
  stated anywhere in the paper.
- **Suggested fix**: Low priority — unlikely to materially affect results at
  this scale; deprioritize relative to the data-mismatch issue.

*A fifth candidate — the LR schedule (`constant`, SIR implementation_assumptions[1], confidence 0.3, the lowest confidence value anywhere in the SIR) — was considered but not flagged as a discrete finding since "constant + early stopping" is a defensible default absent any stated schedule, rather than a specific wrong guess.*

---

## Omission Hallucinations

Components present in the SIR but absent or substituted in the generated code.

### 1. SARSIM-derived Leiden target cluster count K

- **SIR location**: Appendix H "Downstream analysis"; SIR `ambiguities[4]`, confidence 0.5.
- **Severity**: Minor
- **Note**: This is a *substitution*, not a silent omission — `evaluate.py`
  clearly implements an independent Leiden/KMeans resolution sweep with
  silhouette-based K selection, and the code/README both document why (the
  companion SARSIM paper's clustering metadata is not public). Flagged here
  for completeness since it is a component the SIR notes as present in the
  paper's pipeline that could not be faithfully reproduced.
- **Suggested fix**: If SARSIM cluster assignments become available, wire
  them in directly instead of the resolution sweep.

*No Critical omissions were found* — every loss term (Eq. 6/Huber-variant, 8, 9, 10), every architectural module, and the full evaluation metric suite (ARI, NMI, spatial contiguity, silhouette, MUS) from the SIR are implemented in the generated code.

---

## Summary

| Type | Count | Critical | Significant | Minor |
|------|-------|----------|--------------|-------|
| Structural | 1 | 0 | 0 | 1 |
| Parametric | 4 | 0 | 1 | 3 |
| Omission | 1 | 0 | 0 | 1 |

No Critical-severity hallucinations were identified. All flagged items trace
directly back to documented SIR ambiguities or implementation assumptions
(each already carrying its own confidence score) rather than undisclosed
deviations introduced during code generation.
