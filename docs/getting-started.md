# Getting started with ArXivist

## Prerequisites

- Access to Claude (claude.ai, Claude API, or Claude Code)
- The ArXivist skill installed (see below)
- A paper to process: an arXiv URL, DOI, or PDF

---

## Installing the skill

### Option 1 — Clone the repository

```bash
git clone https://github.com/qosi-org/arxivist.git
cd arxivist
```

Point your Claude skill loader at `skill/SKILL.md`. The exact method depends on your
Claude environment — refer to your platform's skill installation documentation.

### Option 2 — Download a release

Download the latest `arxivist-vX.Y.Z.skill` from the
[Releases page](https://github.com/qosi-org/arxivist/releases) and install it via your
platform's skill import mechanism.

---

## Your first run

Start a Claude conversation with the skill active and say:

```
Use ArXivist to implement this paper: https://arxiv.org/abs/1706.03762
```

ArXivist will:

1. Detect the paper (arXiv:1706.03762 — "Attention Is All You Need")
2. Generate a `paper_id`: `arxiv_1706_03762`
3. Run Stage 1 — parse the paper into a SIR
4. Run Stage 2 — commit the SIR to the registry
5. Run Stage 3 — produce an architecture plan
6. Run Stage 4 — generate the full repository
7. Run Stage 5 — produce a Jupyter notebook

After each stage you will see a status block:

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

---

## Workspace layout after a run

```
workspace/
├── sir-registry/
│   ├── global_index.json              ← Updated with your paper
│   └── arxiv_1706_03762/
│       ├── sir.json                   ← Full SIR
│       ├── metadata.json
│       ├── pipeline_state.json
│       ├── architecture_plan.json
│       ├── architecture_plan_summary.md
│       └── versions/
│           └── sir_v1.json
│
└── paper-repos/
    └── arxiv_1706_03762/
        ├── src/transformer/           ← Generated source code
        ├── configs/config.yaml
        ├── docker/Dockerfile
        ├── data/download.sh
        ├── notebooks/reproduce_arxiv_1706_03762.ipynb
        ├── checkpoints/
        ├── results/
        ├── comparison/
        └── README.md
```

---

## Running the generated code

```bash
cd workspace/paper-repos/arxiv_1706_03762

# Install
pip install -e .

# Download data
bash data/download.sh

# Quick debug run (reduced config, synthetic data)
python train.py --config configs/config.yaml --debug

# Full training
python train.py --config configs/config.yaml

# Open the notebook
jupyter notebook notebooks/reproduce_arxiv_1706_03762.ipynb
```

---

## Running Stage 6 — Results Comparator

After you have trained the model and have results, tell ArXivist:

```
I ran the generated Transformer code and got BLEU 27.1 on WMT14 EN-DE test set
after 100k steps. Compare against the paper.
```

ArXivist runs Stage 6 and writes four comparison artifacts to
`workspace/paper-repos/arxiv_1706_03762/comparison/`.

---

## Resuming an interrupted run

If a run is interrupted at any stage, tell ArXivist:

```
Resume the ArXivist pipeline for arxiv_1706_03762
```

It loads `pipeline_state.json` and continues from the last completed stage.

---

## Processing multiple papers

Each paper gets its own `paper_id` and its own isolated folder in `sir-registry/` and
`paper-repos/`. The `global_index.json` accumulates an entry for every processed paper,
building a persistent scientific memory across all your runs.
