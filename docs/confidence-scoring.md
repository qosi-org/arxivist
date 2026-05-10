# Confidence scoring

Every section of every SIR carries a confidence score between 0.0 and 1.0. These scores
flow through the entire pipeline, determining how warnings are surfaced, whether the pipeline
pauses, and how generated code is annotated.

---

## Score definitions

| Range | Meaning | Example |
|-------|---------|---------|
| 0.9–1.0 | Explicitly stated in the paper | Optimiser named as "Adam" with exact β₁, β₂ values in a table |
| 0.7–0.89 | Strongly implied or standard practice | "We use the standard learning rate schedule" with warmup mentioned |
| 0.5–0.69 | Inferred with reasoning | Batch size estimated from GPU count × typical memory usage |
| < 0.5 | Speculative — must be reviewed | Architecture detail only visible in a low-resolution figure |

Scores should never be rounded to 0.0 or 1.0 unless the information is completely absent
(0.0) or verbatim-confirmed with no ambiguity at all (1.0 is exceedingly rare).

---

## Where confidence scores appear

### In the SIR

Every module, equation, tensor, training field, and evaluation field has a `confidence` field.
The top-level `confidence_annotations` object summarises confidence per section.

### In the architecture plan

Low-confidence SIR sections trigger risk entries in the architecture plan's `risk_assessment[]`.
The generated `config.yaml` annotates every low-confidence hyperparameter:

```yaml
# ASSUMED: xavier uniform init — paper does not specify (confidence: 0.70)
weight_init: xavier_uniform
```

### In generated code

Every assumed hyperparameter carries a `# ASSUMED: <basis>` comment inline.
Components derived from SIR sections with confidence < 0.6 are implemented as stubs:

```python
class UnknownComponent(nn.Module):
    """
    STUB: Architecture not described in sufficient detail.
    SIR ambiguity: Head count unclear — text says 'multi-head' without specifying H.
    Replace this stub before training.
    """
    def forward(self, x):
        raise NotImplementedError("See docstring — requires manual implementation")
```

---

## Pipeline behaviour by confidence level

| Condition | Pipeline action |
|-----------|----------------|
| Section confidence ≥ 0.7 | Proceed; no warning |
| Section confidence 0.5–0.69 | Surface warning to user; continue |
| Section confidence < 0.5 | Set `human_review_required: true`; pause and wait for confirmation |
| Overall SIR confidence < 0.65 | Pause the entire pipeline and request human review |

---

## Confidence in the results comparator

Stage 6 uses confidence data from the SIR when performing root cause analysis. A large
deviation on a metric that corresponds to a low-confidence SIR training section is more
likely to be caused by an incorrect assumption than by a hallucination.

The reproducibility score itself carries a `score_confidence` field (`high`, `medium`, `low`)
that reflects how many metrics were matched and how complete the user's results are.

---

## Improving low-confidence sections

If you have domain knowledge that resolves an ambiguity, you can:

1. Edit `workspace/sir-registry/{paper_id}/sir.json` directly to update the relevant field
   and raise its confidence score.
2. Tell ArXivist: "Resume the pipeline for {paper_id} from Stage 3" — it will reload the
   updated SIR and re-plan accordingly.
3. The registry will increment the SIR version and preserve the previous version.
