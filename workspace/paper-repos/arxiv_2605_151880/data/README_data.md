# Data Setup for FutureSim

FutureSim requires two data assets:

---

## 1. CCNews Article Corpus

The paper uses a **7.36M-article snapshot** of [Common Crawl News (CCNews)](https://data.commoncrawl.org/crawl-data/CC-NEWS/index.html)
spanning January 2023 – March 2026, from 141 distinct sources.

### Directory structure expected by FutureSim

```
data/ccnews/
├── 2023/
│   ├── 01/
│   │   ├── 01/
│   │   │   └── articles.jsonl
│   │   └── ...
│   └── ...
├── 2024/
│   └── ...
└── 2026/
    └── 03/
        └── 28/
            └── articles.jsonl
```

Each `articles.jsonl` file has one JSON object per line:
```json
{"text": "Article body...", "url": "https://...", "pub_date": "2026-01-15", "source": "Al Jazeera"}
```

### Download from Common Crawl

```bash
# List available CC-NEWS crawls
curl https://data.commoncrawl.org/crawl-data/CC-NEWS/index.html

# Download and extract (example for one month)
aws s3 cp s3://commoncrawl/crawl-data/CC-NEWS/2026/01/ . --recursive --no-sign-request
```

**Note**: The full corpus is very large (~hundreds of GB). For testing, create a small
synthetic subset using the format above.

---

## 2. Forecasting Questions CSV

The paper's 330 questions (Al Jazeera, Q1 2026) are generated using `generate_questions.py`.
To regenerate:

```bash
python generate_questions.py \
    --articles-path data/ccnews/ \
    --output-csv data/questions.csv \
    --model gpt-4o \
    --n-questions 500
```

Expected columns in `data/questions.csv`:
```
qid, title, background, resolution_criteria, answer_type,
resolution_date, ground_truth, source_url, source_pub_date
```

---

## 3. Synthetic Test Data (no download required)

To test the pipeline without downloading CCNews, create a minimal corpus:

```python
import json, pathlib, datetime

for day in range(5):
    d = datetime.date(2026, 1, 1) + datetime.timedelta(days=day)
    path = pathlib.Path(f"data/ccnews/{d.year}/{d.month:02d}/{d.day:02d}")
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "articles.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({
                "text": f"Synthetic article {i} for {d}. Some world event happened today.",
                "url": f"https://example.com/{d}/{i}",
                "pub_date": str(d),
                "source": "synthetic",
            }) + "\n")
print("Synthetic corpus created.")
```
