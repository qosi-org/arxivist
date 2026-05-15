# FutureSim: Replaying World Events to Evaluate Adaptive Agents

**ArXivist-generated reproduction repository**  
Paper: [arXiv:2605.15188](https://arxiv.org/abs/2605.15188)  
Authors: Shashwat Goel, Nikhil Chandak, Arvindh Arun, Ameya Prabhu, Steffen Staab, Moritz Hardt, Maksym Andriushchenko, Jonas Geiping  

---

## What This Paper Does

FutureSim is a benchmarking framework that evaluates how well AI agents adapt to new information over long time horizons. Rather than using simulated or synthetic environments, FutureSim *replays real-world events* chronologically: each day, agents can read real news articles that have just "arrived" and must forecast world events beyond their knowledge cutoff.

Agents interact with the environment through just two actions:
- `submit_forecast(question_id, outcomes)` — submit a probability distribution over free-form outcomes
- `next_day()` — advance the simulation by one day

**Key finding**: The best frontier agent (GPT 5.5) achieves only **25% accuracy** on 330 real-world forecasting questions over Jan–Mar 2026. Three of five tested open-weight models score below zero on the Brier Skill Score — meaning they'd do better by abstaining entirely.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd futuresim-arxiv_2605_151880
pip install -e .

# 2. Create synthetic test data (no CCNews download needed)
python -c "
import json, pathlib, datetime
for day in range(5):
    d = datetime.date(2026, 1, 1) + datetime.timedelta(days=day)
    p = pathlib.Path(f'data/ccnews/{d.year}/{d.month:02d}/{d.day:02d}')
    p.mkdir(parents=True, exist_ok=True)
    with open(p / 'articles.jsonl', 'w') as f:
        for i in range(5):
            f.write(json.dumps({'text': f'Event {i} on {d}.', 'url': f'https://ex.com/{i}', 'pub_date': str(d), 'source': 'synthetic'}) + chr(10))
"

# 3. Dry-run to validate setup
python run_simulation.py --config configs/config.yaml --dry-run

# 4. Run full simulation (requires LLM API key and CCNews corpus)
export OPENAI_API_KEY=...
python run_simulation.py --config configs/config.yaml --agent native --model gpt-4o --seed 0
```

---

## Installation

### pip
```bash
pip install -e .
# With dev tools:
pip install -r requirements-dev.txt
```

### conda
```bash
conda create -n futuresim python=3.10
conda activate futuresim
pip install -e .
```

### Docker
```bash
docker-compose up build_index   # Build LanceDB index first
docker-compose up simulation    # Run the simulation
docker-compose up notebook      # Launch Jupyter at localhost:8888
```

---

## Data Setup

See `data/README_data.md` for full instructions. In brief:

1. **CCNews corpus** — download from [Common Crawl](https://data.commoncrawl.org/crawl-data/CC-NEWS/index.html) and organize as `data/ccnews/YYYY/MM/DD/articles.jsonl`
2. **Build search index** — `python build_index.py --corpus-path data/ccnews/ --index-path data/lancedb_index/`
3. **Generate questions** — `python generate_questions.py --articles-path data/ccnews/ --output-csv data/questions.csv`

---

## Usage

### Run simulation
```bash
python run_simulation.py \
    --config configs/config.yaml \
    --agent native \        # or: custom
    --model gpt-4o \
    --seed 0 \
    --output-dir results/
```

### Score saved predictions
```bash
python score_results.py \
    --predictions results/predictions.json \
    --ground-truth data/ground_truth.json \
    --output results/metrics.json
```

### Build index
```bash
python build_index.py \
    --corpus-path data/ccnews/ \
    --index-path data/lancedb_index/ \
    --embedding-model Qwen/Qwen3-Embedding-8B
```

### Generate questions
```bash
python generate_questions.py \
    --articles-path data/ccnews/ \
    --output-csv data/questions.csv \
    --model gpt-4o \
    --n-questions 500
```

---

## Expected Results

From the paper (Section 4.2, Figure 1), recommended harnesses, 3-seed mean:

| Model | Harness | Accuracy | Brier Skill Score |
|---|---|---|---|
| GPT 5.5 | Codex | 25% | +0.05 |
| Claude Opus 4.6 | Claude Code | 20% | +0.02 |
| DeepSeek V4 Pro | Claude Code | 13% | -0.02 |
| GLM 5.1 | Claude Code | 10% | -0.01 |
| Qwen 3.6 Plus | OpenCode | 5% | -0.07 |

---

## Repository Structure

```
├── src/futuresim/
│   ├── environment/
│   │   ├── sim_engine.py       ← Core simulation loop
│   │   ├── task_manager.py     ← CSV question/prediction state
│   │   └── sandbox.py          ← bwrap agent isolation
│   ├── corpus/
│   │   ├── retrieval.py        ← LanceDB hybrid search
│   │   └── ccnews_loader.py    ← CCNews JSONL iterator
│   ├── scoring/
│   │   ├── brier.py            ← BSS, accuracy, TV distance (exact paper equations)
│   │   ├── answer_matcher.py   ← LLM semantic equivalence (Appendix E.5 prompts)
│   │   └── metrics.py          ← Aggregate reporting
│   ├── agents/
│   │   ├── base_agent.py       ← Abstract agent + harness prompts (Appendix E.1, E.2)
│   │   └── harness_multiagent.py ← Multi-agent extension (Appendix E.3)
│   ├── question_gen/
│   │   └── generator.py        ← Question synthesis + resolution date repair (Appendix A, E.4)
│   └── utils/
│       ├── config.py            ← Config dataclasses + seed utilities
│       └── logging.py
├── run_simulation.py           ← Main simulation entrypoint
├── build_index.py              ← CCNews index builder
├── generate_questions.py       ← Question dataset generator
├── score_results.py            ← Offline scoring
├── configs/config.yaml         ← Full configuration with CONFIRMED/ASSUMED annotations
├── docker/Dockerfile
├── docker/docker-compose.yml
├── notebooks/reproduce_arxiv_2605_151880.ipynb
└── data/README_data.md
```

---

## Reproducibility Notes

The following aspects required assumptions beyond what the paper states explicitly:

| Component | Status | Note |
|---|---|---|
| BSS formula | ✅ CONFIRMED | Exactly matches Section 3 + Appendix C |
| Harness prompts | ✅ CONFIRMED | Verbatim from Appendix E.1, E.2 |
| CCNews format | ✅ CONFIRMED | From Appendix E.1 (articles/YYYY/MM/DD/articles.jsonl) |
| Chunk size (512) | ✅ CONFIRMED | Section 4.1 |
| Qwen3-8B embedder | ✅ CONFIRMED | Section 4.1 |
| Hybrid retrieval fusion | ⚠️ ASSUMED | Paper says "hybrid" but doesn't specify fusion method |
| Embedding dimension | ⚠️ ASSUMED | Likely 4096 for Qwen3-8B; not stated |
| bwrap full config | ⚠️ ASSUMED | High-level rules given; specific flags inferred |
| max_actions per day | ⚠️ ASSUMED | Not specified; set to 200 |

---

## Citation

```bibtex
@article{goel2026futuresim,
  title={FutureSim: Replaying World Events to Evaluate Adaptive Agents},
  author={Goel, Shashwat and Chandak, Nikhil and Arun, Arvindh and Prabhu, Ameya and
          Staab, Steffen and Hardt, Moritz and Andriushchenko, Maksym and Geiping, Jonas},
  journal={arXiv preprint arXiv:2605.15188},
  year={2026}
}
```
