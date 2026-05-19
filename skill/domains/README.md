# ArXivist Domain Layer

The `domains/` folder contains subject-domain-specific instruction files that enrich each
pipeline stage with field-specific knowledge. The domain layer is **additive** — it never
replaces a generic agent, it supplements it.

---

## How it activates

After Stage 1 completes, the orchestrator reads `domain_registry.json` and looks up the
paper's detected subject domain from `sir.provenance.subject_domain`. If a matching domain
folder exists, the domain layer is activated for all subsequent stages.

Before each stage runs, the orchestrator loads both:
1. The generic agent file (`agents/0N_*.md`)
2. The domain-specific enrichment file (`domains/{domain}/{file}.md`)

The domain file is loaded **after** the generic agent file. Domain instructions take
precedence over generic ones where they conflict.

If the domain is unrecognised or detection confidence is below 0.6, the domain layer is
skipped and the generic agents run unmodified.

---

## Domain detection

Stage 1 detects the subject domain from:
- Explicit field statements in the abstract or introduction
- Keyword matching against `domain_registry.json → domains[].keywords`
- Journal/conference venue if mentioned (NeurIPS → AI/ML, ICAIF → Finance, etc.)

The detected domain is stored in `sir.provenance.subject_domain` and
`sir.provenance.subject_domain_confidence`.

---

## File structure per domain

Each domain folder contains exactly five files:

| File | Loaded at stage | Purpose |
|---|---|---|
| `parsing_hints.md` | Stage 1 | Field-specific extraction rules for the paper parser |
| `architecture_patterns.md` | Stage 3 | Known module patterns common in this domain |
| `code_conventions.md` | Stage 4 | Domain-specific coding standards and library choices |
| `evaluation_standards.md` | Stage 6 | Metric definitions, acceptable tolerances, comparison norms |
| `data_pitfalls.md` | Stage 1 + 4 | Data-level reproducibility traps specific to this field |

---

## Available domains

| Folder | Subject area |
|---|---|
| `ai/` | Deep learning, foundation models, generative AI |
| `ml/` | Classical ML, probabilistic models, Bayesian methods |
| `finance/` | Quantitative finance, asset pricing, algorithmic trading |
| `economics/` | Econometrics, causal inference, structural models |
| `quantum/` | Quantum computing, variational circuits, quantum ML |
| `biology/` | Bioinformatics, genomics, protein structure |
| `physics/` | Simulation, PDEs, scientific computing |
| `neuroscience/` | Computational neuroscience, BCI, neural decoding |

---

## Adding a new domain

1. Create a new folder under `domains/`.
2. Add all five required files.
3. Add an entry to `domain_registry.json` with the domain label, keywords, file paths,
   and stage enrichment map.
4. Update `sir_schema.json` to include the new domain label in the `subject_domain` enum.
