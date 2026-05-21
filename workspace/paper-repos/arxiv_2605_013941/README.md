# EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents

**Paper:** [arXiv:2605.13941](https://arxiv.org/abs/2605.13941)  
**Authors:** Jiaqi Liu, Xinyu Ye, Peng Xia, Zeyu Zheng, Cihang Xie, Mingyu Ding, Huaxiu Yao  
**Institutions:** UNC-Chapel Hill, UC Berkeley, UCSC  
**Code (original):** https://github.com/aiming-lab/SimpleMem

---

## What This Paper Does

EVOLVEMEM is a memory architecture for LLM agents that does something previous systems don't: it evolves not just *what* it remembers, but *how* it retrieves. Existing memory systems (MemGPT, Mem0, SimpleMem, etc.) freeze their retrieval infrastructure at deployment — the stored content grows, but the scoring functions and fusion strategies stay fixed.

EVOLVEMEM exposes its entire retrieval configuration (fusion mode, top-k budgets, answer styles, per-category overrides) as a structured action space, then runs an LLM-powered diagnosis loop that reads per-question failure logs, identifies root causes, and proposes targeted configuration adjustments. Harmful proposals are automatically reverted; stagnating runs get random perturbations. This closed-loop process — called AutoResearch — starts from a minimal BM25-only baseline and converges to configurations including entirely new dimensions (entity-swap, query decomposition, answer verification) that the system discovered autonomously.

**Results:** +25.7% relative over the strongest baseline on LoCoMo; +18.9% on MemBench; evolved configs transfer with positive (not catastrophic) transfer across benchmarks.

---

## Quick Start

```bash
# 1. Clone and install
git clone <this-repo>
cd evolvemem
pip install -e .
python -m nltk.downloader punkt

# 2. Set API key
export OPENAI_API_KEY=sk-...

# 3. Ingest conversation sessions
python ingest.py --sessions-file data/locomo_sessions.jsonl --db-path evolvemem.db

# 4. Run self-evolution (this is the main contribution)
python evolve.py --qa-file data/locomo_qa.jsonl --db-path evolvemem.db \
                 --output-config outputs/best_config.json

# 5. Evaluate with evolved config
python evaluate.py --qa-file data/locomo_qa.jsonl --db-path evolvemem.db \
                   --evolved-config outputs/best_config.json --output-dir outputs/
```

---

## Installation

### pip
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### conda
```bash
conda create -n evolvemem python=3.11
conda activate evolvemem
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### Docker
```bash
cd docker
docker compose build
docker compose run evolvemem python evolve.py --help
```

---

## Data

See `data/README_data.md` for download instructions for LoCoMo and MemBench.

**Session format** (input to `ingest.py`): JSONL, one session per line, each session a list of `{"speaker": ..., "text": ...}` dicts.

**QA format** (input to `evaluate.py` and `evolve.py`): JSONL, one QA pair per line: `{"q": ..., "ref": ..., "category": ...}`.

---

## Configuration

All parameters are in `configs/config.yaml`. Key sections:

| Section | Key parameters |
|---------|---------------|
| `memory` | `embedding_model`, `window_size=40`, `tau_j=0.80`, `alpha_d=0.05` |
| `retrieval_initial` | `keyword_top_k=5`, `max_context=8`, `fusion_mode=sum` (minimal theta_0) |
| `evolution` | `max_rounds=7`, `epsilon=0.005`, `tau_rev=0.01` |
| `llm` | `provider`, `answer_model`, `extraction_model` |

---

## Expected Results

Results from the paper (Table 2–3). Running with GPT-4o backbone:

### LoCoMo (token-F1)

| Method | MultiHop | SingleHop | Temporal | OpenDomain | Adversarial | **Overall** |
|--------|----------|-----------|---------|------------|-------------|-------------|
| SimpleMem (best baseline) | 0.318 | 0.195 | 0.235 | 0.402 | 0.802 | 0.432 |
| **EVOLVEMEM** | **0.316** | **0.329** | **0.384** | **0.496** | **0.936** | **0.543** |

### MemBench (accuracy %)

| Method | Recall | Reasoning | Robustness | **Overall** |
|--------|--------|-----------|------------|-------------|
| Best baseline (RecentMem/MemGPT) | 62.5 | 50.0 | 62.5 | 57.1 |
| **EVOLVEMEM** | **87.5** | **66.7** | **50.0** | **67.9** |

---

## Evolution Trajectory

The self-evolution loop produces this trajectory on LoCoMo (R0→R7):

| Round | Change | F1 |
|-------|--------|-----|
| R0 | Baseline: BM25-only, k=5, ctx=8 | 30.5% |
| R1 | Enable semantic + RRF fusion | 35.8% |
| R2 | MMR diversity (REVERTED — regression) | 34.8% |
| R3 | Entity-swap for adversarial category | 37.2% |
| R4 | Per-category answer-style flags | 38.5% |
| R5 | Query decomposition for multi-hop | 38.1% |
| R6 | Inferential subtypes + entity-swap expansion | 45.4% |
| R7 | Answer verification + ctx-budget tuning | **54.3%** |

---

## Reproducibility Notes

### Explicitly stated hyperparameters (high confidence)
- Embedding: `BAAI/bge-base-en-v1.5`, 768-dim, batch_size=32
- Window size W=40, sub-window C=15
- tau_J=0.80, alpha_d=0.05, iota_min=0.15, delta_rho=0.05, rho_max=0.30
- Rmax=7, epsilon=0.005, tau_rev=0.01
- Storage: SQLite 3.35+/FTS5

### Assumed hyperparameters (not specified in paper)
| Parameter | Assumed Value | Basis |
|-----------|--------------|-------|
| `lambda_iota` (Eq.1 importance weight) | 1.0 | Not stated; equal weighting |
| `lambda_r` (Eq.1 recency weight) | 1.0 | Not stated; equal weighting |
| `tau_ver` (verification threshold) | 0.5 | Not stated; midpoint |
| RRF smoothing constant k | 60 | Standard literature default |
| `max_retries` for extraction | 3 | Not stated; common practice |

### Known deviations
- BM25 constants k1=1.5, b=0.75 are standard defaults; SQLite FTS5 uses its own BM25 implementation internally (k1≈1.2, b=0.75), which may differ slightly.
- Entity extraction in entity-swap uses capitalized-word heuristic rather than a full NER model (paper doesn't specify the NER approach).

---

## Citation

```bibtex
@article{liu2026evolvemem,
  title={EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents},
  author={Liu, Jiaqi and Ye, Xinyu and Xia, Peng and Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2605.13941},
  year={2026}
}
```
