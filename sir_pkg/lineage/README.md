# sir/lineage

Build a citation and architectural inheritance graph across all SIRs in the registry.
Produces an interactive HTML visualisation, machine-readable JSON, Graphviz DOT, or a
markdown report.

Requires `sir_diff.py` (at `../diff/`) for inheritance edges. Citation edges work
without it.

## Usage

```bash
# Interactive HTML graph (default)
python sir_lineage.py
python sir_lineage.py --out lineage.html

# Adjust similarity threshold for inheritance edges
python sir_lineage.py --threshold 0.60   # stricter
python sir_lineage.py --threshold 0.35   # looser

# Filter to one domain
python sir_lineage.py --domain Finance
python sir_lineage.py --domain AI --out ai_lineage.html

# Ego graph — papers within N hops of a given paper
python sir_lineage.py --ego arxiv_1706_03762 --hops 2

# Other output formats
python sir_lineage.py --format json   --out lineage.json
python sir_lineage.py --format dot    --out lineage.dot
python sir_lineage.py --format report
```

## Edge types

| Type | Style | Meaning |
|---|---|---|
| Inheritance | Solid arrow | SIR similarity ≥ threshold — one paper likely builds on the other |
| Citation | Dashed arrow | Paper B's text mentions Paper A's title |

## Node size and colour

- **Size** — proportional to overall SIR confidence score
- **Colour** — subject domain (AI=indigo, Finance=green, Biology=red, etc.)
- **Gold ring** — paper has a completed Stage 6 reproducibility comparison

## Rendering the DOT output

```bash
dot -Tsvg lineage.dot -o lineage.svg
dot -Tpng lineage.dot -o lineage.png
```
