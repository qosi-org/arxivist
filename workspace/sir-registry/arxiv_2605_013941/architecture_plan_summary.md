# EVOLVEMEM — Architecture Plan Summary

**Paper:** EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents  
**ArXiv:** 2605.13941 | **Plan version:** 1

---

## Framework Decision

Pure Python 3.10+ system — **no neural training, no GPU required**. The paper explicitly states all operations are CPU-only (Apple M-series used for experiments). The "evolution" is an LLM-driven configuration search loop, not gradient descent.

**Core stack:** `sqlite3` (stdlib) · `sentence-transformers` (BGE embeddings) · `rank-bm25` · `openai`/`anthropic` SDK · `numpy`

---

## Module Map

```
src/evolvemem/
├── evolvemem.py             ← Top-level EvolveMem orchestrator
├── memory/
│   ├── store.py             ← MemoryUnit dataclass + MemoryStore (SQLite/FTS5 + BM25 + embeddings)
│   ├── extractor.py         ← MemoryExtractor (sliding window, retry, chunk-split, coverage verify)
│   └── consolidator.py      ← Consolidator (Jaccard dedup, importance decay, entity reinforcement)
├── retrieval/
│   ├── config.py            ← RetrievalConfig (evolvable action space theta — all params with clamp)
│   ├── retriever.py         ← MultiViewRetriever (BM25 + semantic + structured + fusion + entity-swap + decomposition)
│   └── answer_gen.py        ← AnswerGenerator (style-controlled + second-pass verifier)
├── evolution/
│   ├── engine.py            ← EvolutionEngine (Algorithm 1: EVALUATE–DIAGNOSE–PROPOSE–GUARD)
│   └── diagnosis.py         ← DiagnosisModule (LLM reads failure logs → proposes Delta_theta)
├── evaluation/
│   └── metrics.py           ← Token-level F1, BLEU-1
├── embeddings/
│   └── encoder.py           ← SentenceTransformerEmbedder (BGE 768-dim) + HashingEmbedder (64-dim fallback)
├── llm/
│   └── client.py            ← Unified LLM client (OpenAI / Anthropic)
└── utils/
    └── scope.py             ← Hierarchical scope sigma = user:u | workspace:w | session:s
```

---

## Key Data Flow

### Retrieval (per query)
```
query q
  ├─ BM25(q, kkw)        → [(MemUnit, bm25_score), ...]
  ├─ cosine(embed(q), ksem) → [(MemUnit, cos_sim), ...]
  └─ entity_filter(q, kstr) → [(MemUnit, entity_count), ...]
        ↓ fusion (SUM | WEIGHTED-SUM | RRF)
  fused_candidates
        ↓ + importance + recency + reinforcement  [Eq. 1]
  ranked_candidates[:max_context]
        ↓
  answer_gen(q, context, style)  [Eq. 12]
        ↓ optional second-pass verify  [Eq. 13]
  final_answer
```

### Evolution Loop (Algorithm 1)
```
FOR r in 0..7:
  evaluate all QA pairs → per-question raw log L_r
  diagnosis LLM reads L_r → Delta_theta (structured proposal)
  meta-analyzer:
    IF regression > tau_rev: REVERT to theta_star
    ELIF stagnation 2 rounds: EXPLORE (random perturbation)
    ELSE: APPLY clamp(theta_r + Delta_theta)
  IF f_r > f_star: save theta_star
  IF improvement < epsilon: STOP
```

---

## Configuration (config.yaml skeleton)

```yaml
memory:
  db_path: evolvemem.db
  embedding_model: BAAI/bge-base-en-v1.5  # 768-dim
  embedding_batch_size: 32
  window_size: 40          # turns per extraction window
  sub_window_size: 15      # fallback for context-limit splits
  tau_j: 0.80              # Jaccard dedup threshold
  alpha_d: 0.05            # importance decay rate per day
  iota_min: 0.15           # minimum importance floor
  delta_rho: 0.05          # entity reinforcement increment
  rho_max: 0.30            # entity reinforcement cap

retrieval_initial:          # theta_0 — minimal starting config
  keyword_top_k: 5
  semantic_top_k: 0         # disabled at start
  structured_top_k: 0       # disabled at start
  max_context: 8
  fusion_mode: sum
  enable_entity_swap: false
  enable_query_decomposition: false
  enable_answer_verification: false
  answer_style: concise

evolution:
  max_rounds: 7
  epsilon: 0.005
  tau_rev: 0.01
  rrf_k: 60                # ASSUMED: standard RRF default

llm:
  provider: openai          # or anthropic
  extraction_model: gpt-4o
  answer_model: gpt-4o
  diagnosis_model: gpt-4o
  lambda_iota: 1.0          # ASSUMED: weight for importance in ranking
  lambda_r: 1.0             # ASSUMED: weight for recency in ranking
  tau_ver: 0.5              # ASSUMED: verification confidence threshold
```

---

## Entrypoints

| Script | Purpose |
|--------|---------|
| `ingest.py` | Load conversation sessions into the memory store |
| `evaluate.py` | Run QA evaluation with current config; write per-question log |
| `evolve.py` | Run self-evolution loop; output best config |
| `answer.py` | Interactive single-query inference |

---

## Risks

| Severity | Issue | Mitigation |
|----------|-------|-----------|
| 🔴 High | Diagnosis prompt quality is critical; paper's Appendix F prompts are benchmark-tuned | Implement verbatim from Appendix F; allow config override |
| 🟡 Medium | `lambda_iota`, `lambda_r` unspecified | Default 1.0; expose as config |
| 🟡 Medium | `tau_ver` unspecified | Default 0.5; use 'Unknown' class detection as backup trigger |
| 🟢 Low | `Rretry`, RRF `k` unspecified | Standard defaults (3, 60) |

---

## Docker

Base: `python:3.11-slim` (no CUDA needed)  
System deps: `libsqlite3-dev`  
Install: `pip install -r requirements.txt && python -m nltk.downloader punkt`
