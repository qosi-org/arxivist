# Architecture Plan: FutureSim
**paper_id**: `arxiv_2605_151880`  
**Generated**: 2026-05-15

---

## Framework Selection

**Primary Framework**: Python 3.10+, no neural training required  
**Key Libraries**: LanceDB (vector retrieval), sentence-transformers / Qwen3-8B (embeddings), pandas (task state), bwrap (sandboxing)  
**Rationale**: FutureSim is a simulation environment and benchmarking harness, not a trained model. The codebase manages event replay, agent sandboxing, news retrieval, and scoring.

---

## Module Hierarchy

```
src/futuresim/
├── __init__.py
├── environment/
│   ├── __init__.py
│   ├── sim_engine.py          ← Core simulation loop (next_day, state transitions)
│   ├── task_manager.py        ← CSV-based question/prediction state
│   └── sandbox.py             ← bwrap sandbox orchestration
├── corpus/
│   ├── __init__.py
│   ├── ccnews_loader.py       ← CCNews JSONL ingestion and deduplication
│   ├── retrieval.py           ← LanceDB hybrid search wrapper
│   └── embedder.py            ← Qwen3-8B embedding interface
├── scoring/
│   ├── __init__.py
│   ├── brier.py               ← BSS, time-weighted, peer score computation
│   ├── answer_matcher.py      ← LLM-based semantic answer equivalence
│   └── metrics.py             ← Aggregate metrics and reporting
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          ← Abstract agent interface
│   ├── harness_native.py      ← Native harness prompt templates
│   └── harness_custom.py      ← Custom baseline harness with memory tools
├── question_gen/
│   ├── __init__.py
│   ├── generator.py           ← LLM-based question synthesis from articles
│   ├── leakage_filter.py      ← Resolution date repair + leakage checks
│   └── quality_filter.py      ← Difficulty and format filtering
└── utils/
    ├── __init__.py
    ├── config.py              ← Config loading, seed utilities
    └── logging.py             ← Structured logging
```

---

## Key Data Flows

### Simulation Loop
```
SIMULATION ENGINE
  date: datetime  ← current simulation date
  articles_folder: Path  ← date-gated articles/YYYY/MM/DD/
  market_csv: Path  ← task state file (read-only for agent)

  for each day in simulation_window:
    1. expose articles up to current_date to sandbox
    2. invoke agent process (via subprocess in sandbox)
    3. agent calls submit_forecast() → update market.csv predictions
    4. agent calls next_day() → engine advances date
    5. resolve questions whose resolution_date == current_date
    6. compute BSS for resolved questions
    7. write feedback to state file
```

### Search Path
```
RETRIEVAL
  query: str, from_date: str, to_date: str
  → EmbeddingModel.encode(query) → [1, D] float32
  → LanceDB.hybrid_search(embedding, query_text, date_range)
  → top-5 chunks of 512 tokens  ← returned to agent
```

### Scoring Path
```
SCORING
  prediction: Dict[str, float]  ← {outcome: probability}
  ground_truth: str
  → AnswerMatcher.match(outcome, ground_truth) for each outcome → bool
  → brier_skill_score(prediction, matched_outcomes) → float in [-1, 1]
  → aggregate across questions → mean BSS, accuracy
```

---

## Configuration Schema (config.yaml)

```yaml
simulation:
  start_date: "2025-12-24"
  end_date: "2026-03-28"
  timegap_days: 1                  # days between agent wakeups
  max_outcomes_per_question: 5     # CONFIRMED: paper Section 4.1
  seeds: [0, 1, 2]                 # 3 seeds per paper protocol

corpus:
  ccnews_path: "data/ccnews/"      # path to JSONL article files
  index_path: "data/lancedb_index/"
  chunk_size: 512                   # tokens per chunk (CONFIRMED)
  chunks_per_query: 5               # top-k returned (CONFIRMED)
  embedding_model: "Qwen/Qwen3-Embedding-8B"
  # ASSUMED: embedding_dim=4096 — not stated in paper

questions:
  questions_path: "data/questions.csv"
  source: "Al Jazeera / CCNews"
  answer_matcher_model: "deepseek-v3"   # DeepSeek V3.2

scoring:
  metric: "brier_skill_score"
  also_compute: ["accuracy", "time_weighted", "peer_score"]

sandbox:
  use_bwrap: true
  # ASSUMED: standard namespace isolation; full config not in paper
  block_network: true
  article_mount_prefix: "articles/"

harness:
  type: "native"    # or "custom"
  max_actions: 200  # ASSUMED: not specified; reasonable for daily cadence
  max_total_tokens: 200000  # ASSUMED: context window management

hardware:
  device: "cuda"
  embedding_gpu: true
  num_workers: 4
```

---

## Dependencies

### requirements.txt
```
lancedb>=0.6.0
pandas>=2.0.0
numpy>=1.26.0
torch>=2.1.0
transformers>=4.40.0
sentence-transformers>=3.0.0
anthropic>=0.25.0
openai>=1.30.0
tantivy>=0.21.0        # LanceDB full-text search backend
pyarrow>=14.0.0
tqdm>=4.66.0
pyyaml>=6.0.0
python-dateutil>=2.9.0
```

### requirements-dev.txt
```
pytest>=8.0.0
black>=24.0.0
ruff>=0.4.0
mypy>=1.9.0
jupyter>=1.0.0
ipywidgets>=8.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

## Entrypoints

### `run_simulation.py`
```
--config        Path to config YAML
--agent         Agent type: native | custom | external
--model         Model name (e.g. gpt-5.5, claude-opus-4-6)
--seed          Random seed
--debug         Use 10 questions and 5 simulation days
--dry-run       Initialize all components, don't execute
--resume        Path to checkpoint state to resume from
```

### `build_index.py`
```
--corpus-path   Path to CCNews JSONL files
--index-path    Output LanceDB index path
--embedding-model   HuggingFace model ID
--force         Rebuild index even if it exists
```

### `generate_questions.py`
```
--articles-path  Path to source article JSONL files
--output-csv     Output questions CSV
--model          LLM for generation (default: gpt-5.5)
--n-questions    Target number of questions
```

### `score_results.py`
```
--predictions    Path to predictions CSV/JSON
--ground-truth   Path to ground truth CSV
--output         Output metrics JSON
```

---

## Risk Assessment

| Risk | Severity | Description | Mitigation |
|------|----------|-------------|------------|
| Qwen3-8B embedding dimension | Medium | Not stated; affects LanceDB index schema | Default to 4096; make configurable |
| Hybrid retrieval fusion | Medium | Fusion method for semantic+keyword not specified | Use LanceDB default reciprocal rank fusion; add config flag |
| bwrap sandbox | High | Full bwrap args not provided; platform-specific (Linux only) | Provide Docker fallback; document Linux requirement |
| CCNews access | High | 7.36M article corpus not publicly hosted; large download | Provide download script + subset for testing |
| DeepSeek V3.2 API cost | Medium | Answer matching over 10,000+ queries per run | Cache match results; batch queries |
| Proprietary model APIs | High | GPT 5.5, Claude Opus 4.6 require paid API/plan access | Abstract agent interface; mock agent for testing |
| Resolution date leakage | High | Careless implementation could expose future articles | Date-gating enforced at both filesystem and LanceDB level |
