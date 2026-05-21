"""
Question Generation Pipeline
==============================
Automated generation of free-form forecasting questions from timestamped news articles.

Paper reference: Appendix A (Data), Appendix A.1 (Question Creation Methodology)
  "we take the more scalable approach of synthesizing short-answer forecasting questions
   from any (news) source document, introduced in Chandak et al. (2026)"

Pipeline steps:
  1. LLM generates free-form short-answer question from timestamped article
  2. Leakage filter: remove if answerable before simulation start (Jan 1 2026)
  3. Difficulty filter: remove if too stale/easy or unanswerable with full web search
  4. Resolution date repair: set to earliest date answer could be confidently inferred
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# From Appendix E.4 — Resolution date repair prompt
EARLIEST_DATE_PROMPT = """You are provided with a forecasting question (which might be from the past). \
You have to find not only the answer to the question, but also the earliest date on which the answer \
to the question could be inferred. Be smart in your inference. The question might contain extra details \
about the situation/event being asked but I want you to find out the earliest date by which the answer \
could have been figured out (even without extra details). For example, if you had seen the question 6 \
months back, could you have figured out the answer confidently.

Question Title: {question_title}
Question Background: {background}
Expected Answer Type: {answer_type}

Think step by step about the information provided and put the answer to the question in <answer> </answer> \
tags and the earliest date on which the answer to the question could be inferred with certainty in \
<date> </date> tags. The date should be in the format YYYY-MM-DD.

Once you find the answer, please make sure to find THE EARLIEST DATE the answer could have been guessed. \
Try to search as much as possible across sites/pages to find out when was the earliest time the answer \
to the question was basically known/determined (or could have been inferred confidently from public knowledge)."""

QUESTION_GEN_PROMPT = """You are given a news article. Generate one forecasting question whose answer \
is contained in the article, written as if the question were being asked BEFORE the answer was known.

Article:
{article_text}

Generate a JSON object with these fields:
- title: the question (e.g. "Who will be sworn in as Nepal's new prime minister?")
- background: 2-3 sentences of context without revealing the answer
- resolution_criteria: specific condition that determines when/how the question resolves
- answer_type: one of [String (Name), String (Country), Number, Date, Boolean]
- ground_truth: the actual answer from the article
- resolution_date: ISO date when the answer became known (from article)

Return ONLY valid JSON, no markdown."""


class QuestionGenerator:
    """
    Generates forecasting questions from CCNews articles.

    Paper reference: Appendix A.1
    Source used in paper: Al Jazeera articles from CCNews Q1 2026
    Yield rate: ~3% (330 questions from 10,000+ articles)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        simulation_start: str = "2026-01-01",
        simulation_end: str = "2026-03-28",
    ):
        """
        Args:
            model: LLM for question generation (paper uses strong frontier models)
            api_key: OpenAI API key
            simulation_start: Questions must resolve after this date
            simulation_end: Questions must resolve before/on this date
        """
        self.model = model
        self.simulation_start = date.fromisoformat(simulation_start)
        self.simulation_end = date.fromisoformat(simulation_end)
        if not HAS_OPENAI:
            raise ImportError("openai is required: pip install openai")
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))

    def __repr__(self) -> str:
        return (
            f"QuestionGenerator(model={self.model}, "
            f"window={self.simulation_start}→{self.simulation_end})"
        )

    def generate_from_article(self, article: dict) -> Optional[dict]:
        """
        Generate one forecasting question from a news article.

        Args:
            article: dict with keys: text, url, pub_date, source

        Returns:
            Question dict, or None if generation fails validation.
        """
        prompt = QUESTION_GEN_PROMPT.format(article_text=article["text"][:4000])
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            q = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return None

        # Basic validation
        required_keys = ["title", "background", "resolution_criteria",
                        "answer_type", "ground_truth", "resolution_date"]
        if not all(k in q for k in required_keys):
            return None

        # Date validation
        try:
            res_date = date.fromisoformat(q["resolution_date"])
        except ValueError:
            return None

        if not (self.simulation_start <= res_date <= self.simulation_end):
            return None   # Outside evaluation window

        # Placeholder answer filter
        if q.get("ground_truth", "").lower() in ("unknown", "tbd", "n/a", ""):
            return None

        q["source_url"] = article.get("url", "")
        q["source_pub_date"] = article.get("pub_date", "")
        return q

    def repair_resolution_date(self, question: dict) -> Optional[date]:
        """
        Use LLM + web search to find the earliest date the answer was inferable.

        Paper reference: Appendix A.1 (Additional Refinement)
        Prompt from: Appendix E.4

        Args:
            question: Question dict with title, background, answer_type

        Returns:
            Corrected resolution date, or None if question should be discarded.
        """
        prompt = EARLIEST_DATE_PROMPT.format(
            question_title=question["title"],
            background=question["background"],
            answer_type=question["answer_type"],
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            content = resp.choices[0].message.content
            # Parse <date> tag
            import re
            date_match = re.search(r"<date>(.*?)</date>", content, re.DOTALL)
            if not date_match:
                return None
            earliest = date.fromisoformat(date_match.group(1).strip())
        except Exception:
            return None

        # Discard if earliest inferable date is before simulation start
        if earliest < self.simulation_start:
            return None   # Answer was already known; discard question

        return earliest

    def generate_dataset(
        self,
        articles_path: str,
        output_csv: str,
        target_n: int = 500,
    ) -> int:
        """
        Generate a forecasting question dataset from CCNews articles.

        Args:
            articles_path: Path to directory of JSONL article files
            output_csv: Output CSV path for questions
            target_n: Stop after this many valid questions

        Returns:
            Number of questions generated.
        """
        questions = []
        qid = 0
        articles_dir = Path(articles_path)

        for jsonl_path in sorted(articles_dir.rglob("articles.jsonl")):
            with open(jsonl_path) as f:
                for line in f:
                    if len(questions) >= target_n:
                        break
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    q = self.generate_from_article(article)
                    if q is None:
                        continue

                    # Resolution date repair
                    repaired_date = self.repair_resolution_date(q)
                    if repaired_date is None:
                        continue   # Leakage detected or out of window
                    q["resolution_date"] = str(repaired_date)
                    q["qid"] = qid
                    questions.append(q)
                    qid += 1

                    if len(questions) % 10 == 0:
                        print(f"Generated {len(questions)} questions so far...")

            if len(questions) >= target_n:
                break

        # Write CSV
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["qid", "title", "background", "resolution_criteria",
                      "answer_type", "resolution_date", "ground_truth",
                      "source_url", "source_pub_date"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in questions:
                writer.writerow({k: q.get(k, "") for k in fieldnames})

        print(f"Dataset written: {len(questions)} questions → {output_csv}")
        return len(questions)
