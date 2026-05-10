# SIR Specification

The **Scientific Intermediate Representation (SIR)** is the canonical machine-readable
abstraction of a research paper. It is the central artifact of the ArXivist pipeline —
produced by Stage 1, stored by Stage 2, consumed by Stages 3 and 6.

The authoritative schema is at `skill/schemas/sir_schema.json`. This document explains the
intent and usage of every section.

---

## Top-level structure

```json
{
  "paper_id":                   "arxiv_1706_03762",
  "sir_version":                1,
  "provenance":                 { ... },
  "architecture":               { ... },
  "mathematical_spec":          [ ... ],
  "tensor_semantics":           [ ... ],
  "training_pipeline":          { ... },
  "evaluation_protocol":        { ... },
  "implementation_assumptions": [ ... ],
  "ambiguities":                [ ... ],
  "confidence_annotations":     { ... }
}
```

All nine top-level sections are required. A section with no extractable content uses `null`
values with an explanatory note — it is never omitted.

---

## `provenance`

Tracks the paper's identity and the conditions under which the SIR was produced.

| Field | Type | Notes |
|-------|------|-------|
| `title` | string | Full paper title |
| `authors` | string[] | All authors in paper order |
| `affiliations` | string[] | Author affiliations |
| `arxiv_id` | string\|null | e.g. `"1706.03762"` |
| `doi` | string\|null | |
| `submission_date` | string\|null | ISO date |
| `abstract` | string | Truncated to 500 characters |
| `domain` | enum | `CV`, `NLP`, `RL`, `Audio`, `Tabular`, `Multimodal`, `Other` |
| `key_claims` | string[] | 3–5 claims the paper makes about its own contributions |
| `parsed_at` | string | ISO datetime of SIR generation |
| `arxivist_version` | string | ArXivist version that produced this SIR |

---

## `architecture`

The model or system architecture as a directed graph of named components.

### `modules[]`

Each entry represents one named module from the paper:

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | Exact name from paper (e.g. `"MultiHeadAttention"`) |
| `operation_type` | string | What this module does (e.g. `"scaled_dot_product_attention"`) |
| `input_shape` | string\|null | Shape notation e.g. `"[B, T, D]"` |
| `output_shape` | string\|null | Shape notation |
| `parameters` | object | Key hyperparameters for this module |
| `paper_section` | string\|null | Where in the paper this is described |
| `confidence` | float | 0.0–1.0 |
| `notes` | string\|null | Implementation details and caveats |

### `connections[]`

Directed edges in the architecture graph:

| Field | Type | Notes |
|-------|------|-------|
| `from` | string | Source module name |
| `to` | string | Target module name |
| `tensor_name` | string\|null | Name of the tensor flowing along this edge |

### `variants[]`

For papers describing multiple model sizes (Base / Large / XL):

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | Variant name (e.g. `"Transformer-Big"`) |
| `differences` | object | Hyperparameters that differ from the primary variant |

`primary_variant` names which variant the rest of the SIR describes.

---

## `mathematical_spec[]`

Every equation, loss function, and objective from the paper.

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | Human-readable name (e.g. `"scaled_dot_product_attention"`) |
| `latex` | string | LaTeX source, verbatim from paper where possible |
| `role` | enum | `loss`, `objective`, `attention`, `normalization`, `activation`, `other` |
| `used_in` | enum | `training`, `inference`, `both` |
| `hyperparameters` | string[] | Hyperparameter names appearing in the equation |
| `undefined_symbols` | string[] | Symbols used but not defined in the paper |
| `paper_section` | string\|null | Location in paper |
| `confidence` | float | 0.0–1.0 |

---

## `tensor_semantics[]`

The major tensors flowing through the model.

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | Tensor name (e.g. `"query"`, `"logits"`) |
| `shape_notation` | string | e.g. `"[B, T, D]"` where dims are named |
| `dtype` | string\|null | `"float32"`, `"bfloat16"`, `"int64"`, etc. |
| `tensor_type` | enum | `input`, `output`, `intermediate`, `parameter` |
| `module` | string\|null | Which module produces or consumes this tensor |
| `confidence` | float | 0.0–1.0 |

---

## `training_pipeline`

The full training procedure. Every field is nullable — the SIR records `null` when a value
is not in the paper rather than inventing a default.

Key sub-objects:

- **`optimizer`**: `name`, `learning_rate`, `beta1`, `beta2`, `weight_decay`, `confidence`
- **`lr_schedule`**: `type`, `warmup_steps`, `total_steps`, `confidence`
- **Top-level**: `batch_size`, `effective_batch_size`, `training_steps`, `epochs`,
  `mixed_precision`, `gradient_clipping`, `data_augmentation[]`, `curriculum`, `confidence`

---

## `evaluation_protocol`

| Field | Type | Notes |
|-------|------|-------|
| `datasets[]` | object[] | `name`, `split`, `size`, `publicly_available` |
| `metrics[]` | string[] | Metric names (e.g. `["BLEU", "perplexity"]`) |
| `reported_results[]` | object[] | The paper's results table — `metric`, `dataset`, `split`, `value`, `is_primary` |
| `baselines[]` | string[] | Names of baselines the paper compares against |
| `compute` | string\|null | Hardware description from paper |
| `special_conditions[]` | string[] | Beam search settings, ensembling, TTA, etc. |
| `confidence` | float | 0.0–1.0 |

The `reported_results[]` array is the ground truth that Stage 6 uses for comparison.

---

## `implementation_assumptions[]`

Every implicit decision a faithful implementation must make that the paper does not state:

| Field | Type | Notes |
|-------|------|-------|
| `assumption` | string | What was assumed |
| `basis` | string | Why (literature default / common practice / inferred) |
| `confidence` | float | Confidence that this assumption is correct |
| `alternatives` | any[] | Other plausible choices |
| `affects_section` | string | Which SIR section this assumption affects |

Stage 4 emits every assumption as a `# ASSUMED: <basis>` comment in the generated code.

---

## `ambiguities[]`

Points in the paper where the correct interpretation is genuinely unclear:

| Field | Type | Notes |
|-------|------|-------|
| `location` | string | Paper section or figure reference |
| `description` | string | What is ambiguous |
| `primary_assumption` | any | The most likely interpretation |
| `alternatives` | any[] | Other plausible interpretations |
| `confidence` | float | Confidence in the primary assumption |

Ambiguities with confidence < 0.6 pause the pipeline and request human review.

---

## `confidence_annotations`

Section-level summary scores:

```json
{
  "architecture":                0.97,
  "mathematical_spec":           0.98,
  "tensor_semantics":            0.96,
  "training_pipeline":           0.91,
  "evaluation_protocol":         0.97,
  "implementation_assumptions":  0.82,
  "overall_sir_confidence":      0.94
}
```

The `overall_sir_confidence` is used by the orchestrator to decide whether to pause for
human review (threshold: 0.65) and is stored in `global_index.json` for each paper.

See [Confidence scoring](confidence-scoring.md) for full details.
