# ArXivist Examples

Pre-generated SIR artifacts and architecture plan summaries for well-known papers.
These serve as reference implementations of the ArXivist output format and can be used
to understand what each stage produces before running the skill on a new paper.

---

## Available examples

### Attention Is All You Need (Vaswani et al., 2017)

arXiv: [1706.03762](https://arxiv.org/abs/1706.03762)

| File | Description |
|------|-------------|
| `sir.json` | Complete SIR — architecture graph, 7 equations, full tensor semantics, training pipeline, WMT14 BLEU results, 5 implementation assumptions, 2 ambiguities |
| `architecture_plan_summary.md` | Human-readable architecture plan: module hierarchy, tensor flows, config schema, risk assessment |

**SIR confidence**: 0.94 overall (highest confidence in evaluation protocol at 0.97, lowest in training pipeline at 0.91 due to ambiguous batch size specification)

---

## Adding an example

1. Process a paper through ArXivist Stages 1–3.
2. Copy `workspace/sir-registry/{paper_id}/sir.json` to `examples/{paper-slug}/sir.json`.
3. Copy `workspace/sir-registry/{paper_id}/architecture_plan_summary.md` to `examples/{paper-slug}/`.
4. Add an entry to this README.
5. Validate: `python3 -m json.tool examples/{paper-slug}/sir.json > /dev/null`
6. Open a pull request — CI will validate the SIR against `sir_schema.json` automatically.

Do not commit generated repository code (Stage 4 output) to the examples directory.
Generated repos live in `workspace/paper-repos/` which is gitignored.
