# How ArXivist works

ArXivist is a multi-agent Claude skill. The master orchestrator (`SKILL.md`) reads a paper
and coordinates six specialist sub-agents, each of which owns one well-defined stage of the
pipeline. Sub-agents are loaded one at a time — only the relevant agent's instructions are
in Claude's context during that stage, keeping each stage focused and preventing cross-stage
contamination.

---

## Stage 1 — Paper Parser

**File**: `skill/agents/01_paper_parser.md`
**Input**: PDF, arXiv URL, or raw text
**Output**: `sir-registry/{paper_id}/sir.json`

The Paper Parser reads the paper from cover to cover and extracts a structured Scientific
Intermediate Representation (SIR). It works through eight sections in a fixed order:

1. Provenance and metadata
2. Architecture graph (named modules with tensor shapes and connections)
3. Mathematical specification (all equations in LaTeX)
4. Tensor semantics (shapes, dtypes, roles)
5. Training pipeline (optimiser, schedule, batch size, augmentation)
6. Evaluation protocol (datasets, metrics, results table)
7. Implementation assumptions (everything the paper leaves implicit)
8. Confidence annotations (per-section scores from 0.0 to 1.0)

Every ambiguity is recorded explicitly in `ambiguities[]` rather than silently guessed.
Every assumption is recorded in `implementation_assumptions[]` with a basis and alternatives.

See [SIR specification](sir-specification.md) for the full format.

---

## Stage 2 — SIR Registry

**File**: `skill/agents/02_sir_registry.md`
**Input**: SIR artifact from Stage 1
**Output**: Updated registry files + `global_index.json`

The registry keeper commits the SIR to `workspace/sir-registry/` with versioning and
provenance tracking. The registry is append-only — every version of every SIR is retained.
`global_index.json` accumulates one entry per paper, building a persistent scientific memory.

Only the orchestrator writes to the registry. Sub-agents return artifacts; the orchestrator
commits them.

---

## Stage 3 — Architecture Planner

**File**: `skill/agents/03_architecture_planner.md`
**Input**: SIR from the registry
**Output**: `sir-registry/{paper_id}/architecture_plan.json` + `architecture_plan_summary.md`

The Architecture Planner translates the SIR's abstract architecture graph into a concrete
software plan. It produces:

- Framework selection with reasoning
- Complete Python module hierarchy (every file, every class, public method signatures)
- Tensor flow pseudocode for every major forward pass
- Full `config.yaml` schema with all hyperparameters and confidence comments
- Dependencies manifest (`requirements.txt`, `requirements-dev.txt`, `environment.yaml`)
- Entrypoint CLI schemas (`train.py`, `evaluate.py`, `inference.py`)
- Docker specification
- Risk assessment for all low-confidence SIR sections

The planner does not write code — it produces a blueprint that Stage 4 implements.

---

## Stage 4 — Code Generator

**File**: `skill/agents/04_code_generator.md`
**Input**: Architecture plan + SIR
**Output**: Full repository at `workspace/paper-repos/{paper_id}/`

The Code Generator writes the entire repository following the architecture plan exactly. It
generates files in dependency order (utilities first, models next, training last) and enforces
strict reproducibility rules:

- All random operations are seedable
- All paths are configurable via config — nothing hardcoded
- Tensor shapes are asserted at module boundaries
- Every assumed hyperparameter carries a `# ASSUMED: <basis>` comment
- Components that cannot be faithfully implemented are replaced with clearly-labelled stubs
- The `README.md` includes a "Reproducibility Notes" section documenting all low-confidence
  sections and assumptions from the SIR

---

## Stage 5 — Notebook Generator

**File**: `skill/agents/05_notebook_generator.md`
**Input**: Repo file structure + SIR
**Output**: `workspace/paper-repos/{paper_id}/notebooks/reproduce_{paper_id}.ipynb`

The Notebook Generator produces a Jupyter notebook that is runnable end-to-end on a local
machine without modification. It includes:

- Environment and GPU check
- One-command installation
- Plain-English paper overview with equations rendered in LaTeX
- Forward pass demonstration for each major model component
- Mini training loop on synthetic data (no downloads required)
- Paper results comparison table pulled from the SIR

An optional exploratory notebook is also generated when the paper has interesting
intermediate representations worth visualising.

---

## Stage 6 — Results Comparator (user-triggered)

**File**: `skill/agents/06_results_comparator.md`
**Input**: User's experimental results + SIR
**Output**: Four files in `workspace/paper-repos/{paper_id}/comparison/`

Stage 6 is not run automatically. It is triggered after the user runs the generated code
and provides their results. The comparator:

1. Parses the user's results in any format
2. Matches them against the paper's reported metrics (from the SIR)
3. Computes percentage deviations and classifies severity
4. Computes a reproducibility score (0.0–1.0) with uncertainty estimate
5. Performs root cause analysis for all moderate/significant deviations
6. Audits the generated code for three types of hallucinations: structural (extra components),
   parametric (wrong hyperparameters), and omission (missing components)
7. Writes a full audit trail to `verification_log.md`

---

## Orchestrator validation gates

After each stage the orchestrator validates the output before advancing:

- Stage 1: SIR must conform to `sir_schema.json`; all required fields must be present
- Stage 3: Architecture plan must conform to `architecture_plan_schema.json`
- Stage 6: Comparison report must include reproducibility score, hallucination report, and
  verification log

If validation fails, the orchestrator retries the stage once with targeted prompting. If it
fails again, the pipeline pauses and requests human review.

---

## Pipeline state

Every paper's progress is recorded in `sir-registry/{paper_id}/pipeline_state.json`. This
file is updated after every stage and enables:

- **Resume**: if a run is interrupted, ArXivist picks up from `current_stage`
- **Repair loops**: `loop_count` tracks how many retries have been attempted
- **Audit**: `stage_timestamps` and `stage_confidence` give a full record of when each stage
  ran and how confident the output was
