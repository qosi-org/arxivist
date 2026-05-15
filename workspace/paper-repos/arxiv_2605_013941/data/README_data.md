# Data

## LoCoMo

LoCoMo is publicly available (Section 4.1, Appendix E of the paper).

- **Paper:** "Evaluating very long-term conversational memory of LLM agents" (Maharana et al., ACL 2024)
- **Dataset:** LoCoMo-10 — 10 conversations, 1,986 QA pairs, 5 categories
- **Download:** https://github.com/snap-research/locomo

Expected format after download (`data/locomo/`):
```
data/locomo/
├── conversations/        ← session JSONs
└── qa_pairs.jsonl        ← {"q": ..., "ref": ..., "category": 1-5}
```

Convert to EVOLVEMEM format (one session = list of {"speaker": ..., "text": ...} dicts):
```bash
python data/convert_locomo.py --input data/locomo/ --output data/locomo_sessions.jsonl
```

## MemBench

MemBench is publicly available (Section 4.1, Appendix E of the paper).

- **Paper:** "MemBench: Towards more comprehensive evaluation on the memory of LLM-based agents" (Tan et al., ACL 2025)
- **Download:** https://github.com/memorybench/memorybench

Expected format: 28 samples (7 categories × 2 topics × 2 samples), multiple-choice.

Convert to EVOLVEMEM format:
```bash
python data/convert_membench.py --input data/membench/ --output data/membench_qa.jsonl
```

## QA pair format

All QA files are JSONL with one JSON object per line:

```json
{"q": "What did Alice say about her job?", "ref": "She got promoted to senior engineer.", "category": 2}
```

Fields:
- `q`: Question string
- `ref`: Reference answer string
- `category`: Integer question category (1–5 for LoCoMo; 1–7 for MemBench)
- `conversation_id` (optional): Source conversation identifier
