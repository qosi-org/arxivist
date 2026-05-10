# ArXivist — Complete Onboarding Guide

Everything you need to go from a fresh clone of the repo to a fully running ArXivist pipeline,
and how to feed the outputs back into the right places.

---

## Table of contents

1. [What you need before starting](#1-what-you-need-before-starting)
2. [Clone the repo](#2-clone-the-repo)
3. [Load the skill into Claude](#3-load-the-skill-into-claude)
4. [Run the pipeline on a paper](#4-run-the-pipeline-on-a-paper)
5. [What Claude produces and where it goes](#5-what-claude-produces-and-where-it-goes)
6. [Running the generated code](#6-running-the-generated-code)
7. [Feeding your results back — Stage 6](#7-feeding-your-results-back--stage-6)
8. [Saving everything back into the repo](#8-saving-everything-back-into-the-repo)
9. [Resuming an interrupted run](#9-resuming-an-interrupted-run)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What you need before starting

- **Git** installed
- **Claude access** 
- A paper to process — an arXiv URL (e.g. `https://arxiv.org/abs/1706.03762`), a DOI, or a PDF
- Python 3.10+ and pip (for running the generated code later)
- Docker (optional but recommended for full reproduction)

---

## 2. Clone the repo

```bash
git clone https://github.com/qosi-org/arxivist.git
cd arxivist
```

Your local directory will look like this:

```
arxivist/
├── skill/          ← The ArXivist Claude skill (this is what you give to Claude)
├── workspace/      ← Empty for now — Claude will write everything here
├── docs/
├── examples/
├── README.md
└── CONTRIBUTING.md
```

The `workspace/` folder is where all generated outputs will live. It already has the right
subdirectory structure scaffolded for you — you just need to fill it by running the pipeline.

---

## 3. Load the skill into Claude

The skill is the folder at `skill/`. Claude needs to be able to read this folder during your
conversation. How you do that depends on which Claude interface you are using.

### Option A — claude.ai (recommended for most users)

1. Open [claude.ai](https://claude.ai) and start a new conversation.
2. Upload the following files using the attachment button (paperclip icon). Upload them **all
   at once** so Claude has everything in context:
   - `skill/SKILL.md`
   - `skill/agents/01_paper_parser.md`
   - `skill/agents/02_sir_registry.md`
   - `skill/agents/03_architecture_planner.md`
   - `skill/agents/04_code_generator.md`
   - `skill/agents/05_notebook_generator.md`
   - `skill/agents/06_results_comparator.md`
   - `skill/schemas/sir_schema.json`
   - `skill/schemas/architecture_plan_schema.json`
   - `skill/schemas/comparison_report_schema.json`
   - `skill/templates/sir_template.json`
   - `skill/templates/repo_structure.txt`
   - `skill/templates/comparison_report_template.md`
   - `skill/state/pipeline_state_schema.json`

3. Once uploaded, send this exact message to activate the skill:

   > "I have uploaded the ArXivist skill files. Please read SKILL.md first and confirm you
   > are ready to act as the ArXivist orchestrator."

4. Claude will confirm it has read the master orchestrator and is ready.

> **Note:** Do not start the pipeline yet. Wait for Claude's confirmation before continuing.

### Option B — Claude Code (CLI)

If you are using Claude Code in your terminal:

```bash
cd arxivist
claude --context skill/
```

Claude Code will automatically include all files in the `skill/` directory as context.
Then tell Claude to read `skill/SKILL.md` and confirm readiness before proceeding.

### Option C — Claude API

If you are calling the API directly, include the contents of all skill files in the
`system` prompt or as user-turn document blocks before sending any pipeline instruction.
Refer to the Anthropic API documentation for how to structure multi-document context.

---

## 4. Run the pipeline on a paper

Once Claude has confirmed it is ready, give it a paper. Use any of these formats:

**arXiv URL:**
```
Use ArXivist to implement this paper: https://arxiv.org/abs/1706.03762
```

**PDF upload:**
Upload the PDF first, then say:
```
Use ArXivist to implement the uploaded paper.
```

**By title / DOI:**
```
Use ArXivist to implement this paper: "Attention Is All You Need" by Vaswani et al., 2017
```

Claude will then run Stages 1 through 5 automatically. After each stage it will print a
status block like this:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ArXivist │ Stage 1 Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ SIR produced — 9 modules, 7 equations, confidence 0.94
📁 Written to: workspace/sir-registry/arxiv_1706_03762/sir.json
⚡ Confidence: 0.94
⚠ Review needed: No
Next: Stage 2 — SIR Registry
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**If Claude flags a section for review** (confidence below 0.5), it will pause and describe
the ambiguity. You can either confirm the assumption it made and tell it to continue, or
provide the correct value yourself. Example:

```
The attention head count is ambiguous. Continue with your primary assumption of 8 heads.
```

Let Claude run all five stages before doing anything else.

---

## 5. What Claude produces and where it goes

After Stages 1–5 complete, Claude will have described all the files it generated. You now
need to **create these files locally** by copying Claude's output. Here is exactly what gets
created and where:

### SIR Registry outputs → `workspace/sir-registry/`

```
workspace/sir-registry/
├── global_index.json                          ← Updated by Claude (replace existing file)
└── {paper_id}/                                ← e.g. arxiv_1706_03762/
    ├── sir.json                               ← Full SIR artifact
    ├── metadata.json
    ├── pipeline_state.json
    ├── architecture_plan.json
    ├── architecture_plan_summary.md
    └── versions/
        └── sir_v1.json
```

### Generated repository → `workspace/paper-repos/`

```
workspace/paper-repos/
└── {paper_id}/                                ← e.g. arxiv_1706_03762/
    ├── src/{project_name}/                    ← All source code
    │   ├── models/
    │   ├── data/
    │   ├── training/
    │   ├── evaluation/
    │   └── utils/
    ├── configs/
    │   └── config.yaml
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── data/
    │   └── download.sh
    ├── notebooks/
    │   └── reproduce_{paper_id}.ipynb
    ├── checkpoints/
    ├── results/                               ← You will place your results here later
    ├── comparison/                            ← Stage 6 will write here later
    ├── train.py
    ├── evaluate.py
    ├── inference.py
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── environment.yaml
    └── README.md
```

**How to save Claude's output locally:**

For each file Claude generates, copy the content from the Claude conversation and create the
file at the path it specifies. If Claude generates many files, you can ask it to output them
one at a time:

```
Please output the contents of each generated file one at a time, starting with sir.json.
```

Then create each file locally as Claude produces it.

> **Tip:** If you are using Claude Code, it can write files to disk directly. Tell it:
> "Write all generated files to their specified paths under workspace/."

---

## 6. Running the generated code

Once you have saved all the files locally:

```bash
cd workspace/paper-repos/{paper_id}

# Install dependencies
pip install -e .

# Download the dataset (if publicly available)
bash data/download.sh

# Quick test run (uses reduced config, no real data needed)
python train.py --config configs/config.yaml --debug

# Open the reproduction notebook
jupyter notebook notebooks/reproduce_{paper_id}.ipynb
```

The notebook is the fastest way to verify your setup. It runs a mini training loop on
synthetic data so you do not need to download anything to confirm the code works.

**For full training:**
```bash
python train.py --config configs/config.yaml
```

**For Docker-based runs:**
```bash
cd docker
docker compose up train
```

---

## 7. Feeding your results back — Stage 6

After you have trained the model and have results, return to the same Claude conversation
(do not start a new one — Claude needs the context from the previous stages) and say:

```
I ran the generated code and got the following results:
- BLEU score on WMT14 EN-DE test: 27.1
- Training steps completed: 100,000
- Hardware: RTX 3090
- No config changes were made.

Please run Stage 6 and compare against the paper.
```

Provide as much detail as you can — metric names, dataset splits, training steps, and any
config changes you made. Claude will then produce four comparison artifacts:

| File | What it contains |
|------|-----------------|
| `benchmark_comparison.md` | Side-by-side metric table with deviation percentages |
| `reproducibility_score.json` | Machine-readable score (0.0–1.0) with confidence rating |
| `hallucination_report.md` | Structural, parametric, and omission issues in the generated code |
| `verification_log.md` | Full audit trail of the comparison run |

Save these files to `workspace/paper-repos/{paper_id}/comparison/`.

---

## 8. Saving everything back into the repo

Once Stages 1–6 are complete and all files are saved locally, commit everything:

```bash
cd arxivist

# Stage all workspace outputs
git add workspace/sir-registry/
git add workspace/paper-repos/

# Commit
git commit -m "feat: ArXivist run — {paper_id}

- SIR confidence: {overall_sir_confidence}
- Reproducibility score: {reproducibility_score}
- Stages completed: 1–6"

git push origin main
```

> **What gets committed vs ignored:**
> The `.gitignore` is set up so that the contents of `workspace/` ARE committed — this is
> intentional. Every SIR, generated repo, notebook, and comparison report is part of the
> scientific record and should be version-controlled. The only things gitignored inside
> `workspace/` are build artefacts, Python cache, and checkpoint files.

If you want to share just the SIR for a paper (without the full generated repo), you can
also copy it into `examples/`:

```bash
cp workspace/sir-registry/{paper_id}/sir.json examples/{paper-slug}/sir.json
cp workspace/sir-registry/{paper_id}/architecture_plan_summary.md examples/{paper-slug}/
git add examples/{paper-slug}/
git commit -m "examples: add SIR for {paper title}"
```

---

## 9. Resuming an interrupted run

If a pipeline run stops mid-way (network drop, session timeout, etc.), do not start a new
conversation. Either continue in the same session or start a new one and reload the skill
files. Then tell Claude:

```
Resume the ArXivist pipeline for {paper_id}. The pipeline_state.json shows we completed
Stage 2 and need to continue from Stage 3.
```

Claude will read the pipeline state and continue from where it stopped. If you no longer
have the `pipeline_state.json` content, paste it into the conversation from your local file:

```
Here is the current pipeline_state.json for this paper:
{paste contents}
Please resume from Stage {N}.
```

---

## 10. Troubleshooting

**Claude says it does not know what ArXivist is**
The skill files were not loaded properly. Go back to Step 3 and re-upload all 14 files,
then send the activation message before giving it a paper.

**Stage 1 produces a very low confidence SIR (below 0.65)**
This usually means the paper has limited implementation detail. Claude will pause and ask
for your input on the ambiguous sections. Answer each one and tell it to continue. The
generated code will have more stubs than usual — check the `hallucination_report.md` after
Stage 6 for guidance on what to fill in manually.

**The generated code does not install**
Check `requirements.txt` for version conflicts. You can ask Claude:
```
The pip install failed with this error: {error}. Please fix requirements.txt.
```

**The notebook fails to run**
Return to Claude and say:
```
The notebook fails at cell N with this error: {error}. Please fix it.
```
Claude will repair the affected cells.

**A metric in the comparison has >30% deviation (Critical)**
Read `hallucination_report.md` first — there is likely a structural or parametric issue
in the generated code. The report includes suggested fixes. After fixing, re-run the
relevant training and trigger Stage 6 again in Claude to get an updated comparison.

**I want to reprocess a paper with an improved SIR**
Edit `workspace/sir-registry/{paper_id}/sir.json` to correct any fields, then tell Claude:
```
I have updated the SIR for {paper_id}. Please resume from Stage 3 using the updated SIR.
```
The registry will automatically increment the SIR version and retain the previous one.

---
*ArXivist is built by QOSI.*
