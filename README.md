# ArXivist

**ArXivist** is a multi-agent Claude skill that converts scientific papers into fully executable,
reproducible Git repositories — automatically.

Give it an arXiv URL or a PDF. It reads the paper, extracts a structured Scientific Intermediate
Representation (SIR), plans the software architecture, generates a complete codebase, produces a
runnable Jupyter notebook, and — after you run the code — scores how closely your results match
the paper's reported metrics.

Built by [qosi-org](https://github.com/qosi-org).

---

## How it works

ArXivist orchestrates six specialist sub-agents in sequence:

```
Paper (PDF / arXiv URL)
      │
      ▼
┌─────────────────────┐
│  Stage 1            │  Parses paper into a Scientific Intermediate
│  Paper Parser       │  Representation (SIR) — structured JSON with
│                     │  architecture, equations, training details,
│                     │  evaluation protocol, and confidence scores
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2            │  Commits the SIR to the global registry with
│  SIR Registry       │  versioning, metadata, and provenance tracking
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 3            │  Translates the SIR into a complete software
│  Architecture       │  architecture plan: module hierarchy, tensor
│  Planner            │  flows, configs, dependencies, Docker spec
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 4            │  Generates the full Git repository — source
│  Code Generator     │  code, configs, Dockerfile, dataset scripts,
│                     │  training & evaluation entrypoints, README
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 5            │  Produces a runnable Jupyter notebook that
│  Notebook           │  walks through each component and runs a
│  Generator          │  mini training loop on synthetic data
└────────┬────────────┘
         │
    [you run the code]
         │
         ▼
┌─────────────────────┐
│  Stage 6            │  Compares your results against the paper's
│  Results            │  reported metrics: deviation scores,
│  Comparator         │  hallucination report, reproducibility score
└─────────────────────┘
```

Everything is stored in `workspace/` — a persistent `sir-registry/` of all processed papers, and
a `paper-repos/` folder of generated implementations.

---

## Quickstart

1. Install the ArXivist skill into your Claude environment by pointing it at `skill/SKILL.md`.

2. Start a Claude conversation and say:

   > "Use ArXivist to implement this paper: https://arxiv.org/abs/1706.03762"

3. Claude will run Stages 1–5 automatically, writing all outputs to `workspace/`.

4. Run the generated code. When you have results:

   > "Here are my results — compare them against the paper."

   Claude runs Stage 6 and writes the comparison artifacts into the paper's repository.

---

## Repository structure

```
arxivist/
├── skill/                    # The ArXivist Claude skill
│   ├── SKILL.md              # Master orchestrator (read this first)
│   ├── agents/               # Six specialist sub-agent instruction files
│   ├── schemas/              # JSON schemas for SIR, arch plan, comparison report
│   ├── templates/            # Blank SIR, repo layout, comparison report template
│   └── state/                # Pipeline state schema
│
├── workspace/                # Runtime output directory (contents gitignored)
│   ├── sir-registry/         # Global SIR registry — one folder per paper
│   └── paper-repos/          # Generated paper repositories
│
├── docs/                     # Documentation
├── examples/                 # Pre-generated SIR and arch plan for reference
└── .github/workflows/        # CI — schema validation on every push
```

---

## The SIR format

The Scientific Intermediate Representation (SIR) is the canonical machine-readable abstraction of
a paper. It contains:

- **Provenance** — title, authors, arXiv ID, domain, key claims
- **Architecture graph** — named modules, input/output tensor shapes, connections
- **Mathematical spec** — all equations in LaTeX, named and categorised
- **Tensor semantics** — shapes, dtypes, roles for every major tensor
- **Training pipeline** — optimiser, LR schedule, batch size, augmentation
- **Evaluation protocol** — datasets, metrics, reported results table
- **Implementation assumptions** — everything the paper leaves implicit
- **Ambiguities** — explicitly flagged unclear points with alternatives
- **Confidence annotations** — per-section scores (0.0–1.0)

See [`docs/sir-specification.md`](docs/sir-specification.md) for the full format reference, and
[`skill/schemas/sir_schema.json`](skill/schemas/sir_schema.json) for the JSON schema.

---

## Confidence scoring

Every section of every SIR carries a confidence score:

| Score | Meaning |
|-------|---------|
| 0.9–1.0 | Explicitly stated in the paper |
| 0.7–0.89 | Strongly implied or standard practice |
| 0.5–0.69 | Inferred with reasoning |
| < 0.5 | Speculative — flagged for human review |

Sections below 0.7 surface warnings during generation. Sections below 0.5 pause the pipeline
and require your explicit confirmation before proceeding.

---

## CI

GitHub Actions validates all JSON schemas on every push and pull request.
See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

MIT — see [`LICENSE`](LICENSE).
