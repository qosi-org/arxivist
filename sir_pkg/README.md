# sir/

Standalone tools that operate on the ArXivist SIR registry.

Each tool in this folder reads from `workspace/sir-registry/` and runs independently
of the main skill pipeline. No Claude session required — these are pure Python scripts
you run locally after the pipeline has populated the registry.

---

## Tools

| Folder | What it does |
|---|---|
| `learner/` | Fine-tunes a small language model on accumulated SIR artifacts to build a fast predictive prior for Stage 1 |
| `search/` | Query the registry by natural language or structured filters — find papers by architecture, metric, domain, or training detail |
| `diff/` | Compare two SIRs and produce a structured diff report with a formal similarity score |
| `lineage/` | Build a citation and inheritance graph across all SIRs in the registry |

---

## How they relate

```
workspace/sir-registry/
        │
        ├──▶  learner/   trains on all SIRs → produces a checkpoint
        │                that feeds back into Stage 1 as a prior
        │
        ├──▶  search/    indexes all SIRs → answers queries
        │                ("papers with cross-attention + RoPE")
        │
        ├──▶  diff/      takes any two SIR paths → structured diff
        │                + similarity score (used by lineage)
        │
        └──▶  lineage/   runs diff across all SIR pairs → builds
                         a graph of implementation inheritance
```

## Prerequisites

All tools require Python 3.10+. Individual dependency requirements are in each
tool's own `requirements.txt`. The registry must be populated by at least one
ArXivist pipeline run before any tool here can do useful work.

```bash
# Point each tool at your registry
export ARXIVIST_REGISTRY=workspace/sir-registry/
```
