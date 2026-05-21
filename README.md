# ArXivist

**ArXivist** converts scientific papers into fully executable, reproducible codebases — automatically.

Point it at an arXiv URL, a DOI, or a PDF. ArXivist reads the paper, extracts every architectural
decision, equation, training detail, and evaluation protocol into a structured machine-readable
representation, then generates a complete, runnable repository from it. When you run the code and
have results, it scores how faithfully your implementation reproduces the paper's reported metrics
and tells you exactly why any gaps exist.

Built by [QOSI](https://github.com/qosi-org).

---

## The problem

Reproducing a research paper takes weeks. The gap between reading a paper and having working code
is filled with undocumented hyperparameters, ambiguous architecture descriptions, missing
preprocessing steps, and implementation decisions the authors never wrote down. Most papers are
never reproduced at all.

ArXivist exists to close that gap.

---

## How it works

ArXivist runs a six-stage pipeline, each stage owned by a specialist sub-system:

```
Paper (PDF / arXiv URL / DOI)
         │
         ▼
┌──────────────────────────────────────┐
│  Stage 1 — Paper Parser              │
│                                      │
│  Reads the paper end-to-end and      │
│  extracts a Scientific Intermediate  │
│  Representation (SIR): architecture  │
│  graph, equations in LaTeX, tensor   │
│  shapes, training pipeline, eval     │
│  protocol, and confidence scores.    │
│  Every ambiguity is explicitly       │
│  flagged — nothing is silently       │
│  guessed.                            │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Stage 2 — SIR Registry              │
│                                      │
│  Commits the SIR to a versioned      │
│  global registry with full           │
│  provenance tracking. Every paper    │
│  processed by ArXivist accumulates   │
│  here permanently.                   │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Stage 3 — Architecture Planner      │
│                                      │
│  Translates the SIR into a concrete  │
│  software plan: module hierarchy,    │
│  tensor flow specs, config schema,   │
│  dependency manifest, Docker spec,   │
│  and risk assessment.                │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Stage 4 — Code Generator            │
│                                      │
│  Generates the full repository:      │
│  source code, configs, Dockerfile,   │
│  dataset scripts, training and       │
│  evaluation entrypoints, README.     │
│  Every assumption is annotated.      │
│  Unknown components are stubbed      │
│  and clearly labelled.               │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Stage 5 — Notebook Generator        │
│                                      │
│  Produces a runnable Jupyter         │
│  notebook: component walkthroughs,   │
│  equations rendered inline, and a    │
│  mini training loop on synthetic     │
│  data — no downloads required.       │
└─────────────────┬────────────────────┘
                  │
            [you run the code]
                  │
                  ▼
┌──────────────────────────────────────┐
│  Stage 6 — Results Comparator        │
│                                      │
│  Compares your results against the   │
│  paper's reported metrics. Computes  │
│  a reproducibility score, performs   │
│  root cause analysis on deviations,  │
│  and audits the generated code for   │
│  hallucinations.                     │
└──────────────────────────────────────┘
```

---

## The SIR

At the core of ArXivist is the **Scientific Intermediate Representation** — a structured,
machine-readable abstraction of a paper's entire implementation surface.

The SIR is not a summary. It is a complete engineering specification extracted from prose:

- **Architecture graph** — every named module, its input/output tensor shapes, and the directed connections between them
- **Mathematical spec** — every equation in LaTeX, named, categorised, and linked to its role in training or inference
- **Tensor semantics** — shape notation, dtype, and role for every major tensor
- **Training pipeline** — optimiser, learning rate schedule, batch size, augmentation, mixed precision, gradient clipping
- **Evaluation protocol** — datasets, metrics, and the paper's exact reported results table
- **Implementation assumptions** — every decision the paper leaves implicit, recorded explicitly with a basis and alternatives
- **Ambiguities** — points where the correct interpretation is genuinely unclear, with the most likely interpretation and all alternatives listed
- **Confidence annotations** — per-section scores (0.0–1.0) reflecting how explicitly each detail was stated in the paper

Every SIR is stored permanently in the global registry. As the registry grows, ArXivist
accumulates implementation priors across research — making each successive paper faster
and more accurate to process.

Full format reference: [`docs/sir-specification.md`](docs/sir-specification.md)  
JSON schema: [`skill/schemas/sir_schema.json`](skill/schemas/sir_schema.json)

---

## Confidence scoring

ArXivist never silently guesses. Every extracted detail carries a confidence score:

| Score | Meaning |
|-------|---------|
| 0.9–1.0 | Explicitly stated in the paper |
| 0.7–0.89 | Strongly implied or standard practice |
| 0.5–0.69 | Inferred with reasoning — surfaced as a warning |
| < 0.5 | Speculative — pipeline pauses for human confirmation |

Low-confidence sections propagate forward: they trigger risk entries in the architecture
plan, `# ASSUMED` comments in the generated code, and expanded root cause analysis in the
reproducibility report.

---

## SIR tools

The `sir/` folder contains standalone tools that operate directly on the registry.
No pipeline session required — run them locally after the registry is populated.

**`sir/learner/`** — Fine-tunes a small language model (SmolLM2-360M) on accumulated SIR
artifacts to build a fast predictive prior for Stage 1. Given an abstract, it predicts
likely architecture modules, implementation risks, confidence scores, and ambiguities before
the full PDF is parsed. Improves continuously as the registry grows.

**`sir/search/`** — Query the registry by natural language or structured filters. Find
papers by architecture detail, training configuration, metric, or any SIR field. Uses
TF-IDF by default; upgrades to semantic search if `sentence-transformers` is installed.
Zero required dependencies.

**`sir/diff/`** — Compare any two SIRs and produce a structured diff report with a formal
similarity score (0.0–1.0) weighted across architecture, equations, training pipeline,
evaluation protocol, and tensor semantics. Useful for measuring architectural inheritance
between papers or tracking changes across ArXivist runs.

**`sir/lineage/`** — Builds a citation and inheritance graph across all SIRs in the
registry. Produces an interactive force-directed HTML visualisation, machine-readable JSON,
Graphviz DOT, or a markdown report. Makes the accumulated registry visible as a knowledge
graph.

---

## Reproducibility report

Stage 6 produces four artifacts that constitute a complete scientific audit:

| Artifact | Contents |
|---|---|
| `benchmark_comparison.md` | Metric-by-metric comparison table with deviation percentages and severity ratings |
| `reproducibility_score.json` | Score (0.0–1.0) with confidence estimate and per-metric breakdown |
| `hallucination_report.md` | Structural, parametric, and omission issues in the generated code |
| `verification_log.md` | Full audit trail — timestamps, input hash, SIR version, config changes |

These are committed permanently into the paper's repository alongside the generated code,
forming a complete scientific provenance record for every implementation ArXivist produces.

---

## Repository structure

```
arxivist/
├── skill/                      # ArXivist pipeline — six-stage system
│   ├── SKILL.md                # Master orchestrator
│   ├── agents/                 # Stage instruction files (01–06)
│   ├── schemas/                # SIR, architecture plan, comparison report schemas
│   ├── templates/              # Blank SIR, repo layout, report template
│   └── state/                  # Pipeline state schema
│
├── sir/                        # Standalone SIR tools
│   ├── learner/                # LoRA fine-tuning on SIR corpus
│   ├── search/                 # Registry search (TF-IDF + semantic)
│   ├── diff/                   # Structured SIR diff and similarity scoring
│   └── lineage/                # Inheritance and citation graph
│
├── workspace/                  # Runtime output (gitignored contents)
│   ├── sir-registry/           # Global SIR registry — one folder per paper
│   │   └── global_index.json   # Index of every processed paper
│   └── paper-repos/            # Generated paper repositories
│
├── docs/                       # Documentation
├── examples/                   # Reference SIRs for well-known papers
└── .github/workflows/          # CI — schema validation on every push
```

---

## Quickstart

See [`INSTRUCTIONS.md`](INSTRUCTIONS.md) for the full onboarding walkthrough.

```bash
git clone https://github.com/qosi-org/arxivist.git
cd arxivist
```

Load the `skill/` folder into your ArXivist environment, then:

```
Use ArXivist to implement this paper: https://arxiv.org/abs/1706.03762
```

ArXivist runs Stages 1–5 and writes everything to `workspace/`. Open the generated
notebook to verify your setup, run the full training pipeline, then feed your results back
to trigger Stage 6.

---

## CI

Every push and pull request validates all JSON schemas, confirms all six stage files are
present and correctly structured, and verifies the workspace scaffold and example SIRs
are intact.

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

MIT — see [`LICENSE`](LICENSE).
