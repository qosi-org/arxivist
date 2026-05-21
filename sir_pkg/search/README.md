# sir/search

Query the ArXivist SIR registry using natural language or structured filters.

Works out of the box with no dependencies — uses TF-IDF over all SIR text fields.
Install `sentence-transformers` for semantic search.

## Usage

```bash
# Natural language
python sir_search.py "papers using diffusion with cosine noise schedule"

# Structured filter
python sir_search.py --filter "domain=Finance AND metric=Sharpe"
python sir_search.py --filter "optimizer=Adam AND confidence.architecture>0.85"

# Field report across all papers
python sir_search.py --field training_pipeline.optimizer.name
python sir_search.py --field confidence_annotations.overall_sir_confidence

# Look up a specific paper
python sir_search.py --paper-id arxiv_1706_03762

# JSON output for piping
python sir_search.py "attention mechanism" --json | jq '.[].paper_id'

# Semantic search (requires sentence-transformers)
python sir_search.py "equivariant graph network for molecules" --semantic
```

## Supported filter operators

| Operator | Meaning | Example |
|---|---|---|
| `=` | Exact match | `domain=Finance` |
| `!=` | Not equal | `domain!=AI` |
| `~` | Contains (substring) | `title~diffusion` |
| `>` | Greater than (numeric) | `confidence.overall>0.8` |
| `<` | Less than (numeric) | `batch_size<32` |

Join conditions with `AND`.

## Field aliases

| Short name | Full path |
|---|---|
| `domain` | `provenance.subject_domain` |
| `title` | `provenance.title` |
| `optimizer` | `training_pipeline.optimizer.name` |
| `metric` | `evaluation_protocol.metrics` |
| `module` | `architecture.modules` |
| `confidence` | `confidence_annotations.overall_sir_confidence` |
| `batch_size` | `training_pipeline.batch_size` |
| `framework` | `architecture.primary_variant` |
