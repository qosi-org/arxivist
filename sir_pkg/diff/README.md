# sir/diff

Compare two SIRs and produce a structured diff report with a formal similarity score.

No dependencies beyond Python stdlib.

## Usage

```bash
# Compare two SIR files directly
python sir_diff.py path/to/sir_a.json path/to/sir_b.json

# Compare two papers in the registry by paper_id
python sir_diff.py --id-a arxiv_1706_03762 --id-b arxiv_2005_14165

# Output formats
python sir_diff.py sir_a.json sir_b.json --format markdown   # default
python sir_diff.py sir_a.json sir_b.json --format json
python sir_diff.py sir_a.json sir_b.json --format summary    # one line

# Write to file
python sir_diff.py sir_a.json sir_b.json --out diff_report.md
```

## Similarity score

The overall score (0.0–1.0) is a weighted average across five sections:

| Section | Weight | What it measures |
|---|---|---|
| Architecture | 0.35 | Module name and connection overlap |
| Mathematical spec | 0.20 | Equation name and role overlap |
| Training pipeline | 0.20 | Optimizer, schedule, batch size match |
| Evaluation protocol | 0.15 | Metric and dataset overlap |
| Tensor semantics | 0.10 | Tensor name and shape overlap |

## Interpretation

| Score | Meaning |
|---|---|
| ≥ 0.85 | Nearly identical implementations |
| 0.60–0.84 | Significant architectural inheritance |
| 0.30–0.59 | Same paradigm, different design |
| < 0.30 | Largely independent implementations |
