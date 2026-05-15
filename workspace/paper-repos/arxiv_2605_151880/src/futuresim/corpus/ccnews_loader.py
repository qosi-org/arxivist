"""
CCNews Corpus Loader
======================
Loads and iterates over the CCNews article corpus used in FutureSim.

Paper reference: Section 4.1 (Search Corpus), Appendix B.5 (Context Details: CCNews)
  "The search corpus is a deduplicated snapshot of CCNews containing 7.36M articles
   from 141 distinct news sources between January 2023 and March 2026."
  "Articles are split into 512-token text chunks, embedded with Qwen3 Embedding 8B,
   and indexed for hybrid retrieval."
  Format: articles/YYYY/MM/DD/articles.jsonl
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterator


class CCNewsLoader:
    """
    Iterates over CCNews JSONL articles organized by date.

    Expected directory structure:
      corpus_root/
        YYYY/
          MM/
            DD/
              articles.jsonl    ← one JSON object per line

    Each article JSON:
      {"text": "...", "url": "...", "pub_date": "YYYY-MM-DD", "source": "..."}
    """

    def __init__(self, corpus_root: str):
        self.corpus_root = Path(corpus_root)
        assert self.corpus_root.exists(), f"Corpus root not found: {self.corpus_root}"

    def __repr__(self) -> str:
        return f"CCNewsLoader(root={self.corpus_root})"

    def iter_articles(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> Iterator[dict]:
        """
        Iterate over all articles within the given date range.

        Args:
            from_date: Inclusive start date (None = no lower bound)
            to_date: Inclusive end date (None = no upper bound)

        Yields:
            Article dicts with keys: text, url, pub_date, source
        """
        for year_dir in sorted(self.corpus_root.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                try:
                    month = int(month_dir.name)
                except ValueError:
                    continue

                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    try:
                        day = int(day_dir.name)
                        art_date = date(year, month, day)
                    except (ValueError, TypeError):
                        continue

                    if from_date and art_date < from_date:
                        continue
                    if to_date and art_date > to_date:
                        continue

                    jsonl = day_dir / "articles.jsonl"
                    if not jsonl.exists():
                        continue
                    with open(jsonl) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue

    def count_articles(self, from_date: date | None = None, to_date: date | None = None) -> int:
        """Count articles in date range without loading full text."""
        return sum(1 for _ in self.iter_articles(from_date, to_date))

    def available_dates(self) -> list[date]:
        """Return sorted list of all dates that have article files."""
        dates = []
        for art_path in sorted(self.corpus_root.rglob("articles.jsonl")):
            parts = art_path.parent.parts
            try:
                d = date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
                dates.append(d)
            except (ValueError, IndexError):
                continue
        return sorted(dates)
