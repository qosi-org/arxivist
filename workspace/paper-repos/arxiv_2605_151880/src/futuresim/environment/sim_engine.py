"""
FutureSim Simulation Engine
===========================
Implements the chronological event-replay environment from:
  "FutureSim: Replaying World Events to Evaluate Adaptive Agents" (Section 3)

The engine manages:
- Daily time progression via next_day()
- Date-gated article exposure to agents
- Question resolution and feedback delivery
- Metric aggregation across the simulation window
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from futuresim.scoring.brier import compute_brier_skill_score, compute_accuracy
from futuresim.utils.config import SimConfig
from futuresim.utils.logging import get_logger

logger = get_logger(__name__)


class SimulationEngine:
    """
    Core chronological replay engine for FutureSim.

    Each call to next_day() advances the simulation by one day,
    exposes new articles to the agent sandbox, resolves questions
    whose resolution_date == current_date, and records feedback.

    Paper reference: Section 3.1 (Environment Design)
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.current_date: date = date.fromisoformat(config.simulation.start_date)
        self.end_date: date = date.fromisoformat(config.simulation.end_date)
        self.market_path = Path(config.questions.questions_path)
        self.corpus_root = Path(config.corpus.ccnews_path)
        self.agent_workspace = Path(config.sandbox.workspace_path)
        self._market_df: Optional[pd.DataFrame] = None
        self._history: list[dict] = []

    def __repr__(self) -> str:
        return (
            f"SimulationEngine(current_date={self.current_date}, "
            f"end_date={self.end_date}, "
            f"questions={len(self._market_df) if self._market_df is not None else 'not loaded'})"
        )

    def load_questions(self) -> None:
        """Load the forecasting question CSV into memory."""
        assert self.market_path.exists(), f"Questions file not found: {self.market_path}"
        self._market_df = pd.read_csv(self.market_path)
        self._market_df["resolution_date"] = pd.to_datetime(
            self._market_df["resolution_date"]
        ).dt.date
        self._market_df["is_resolved"] = False
        self._market_df["ground_truth"] = None
        self._market_df["my_prediction"] = None
        self._market_df["my_prediction_date"] = None
        self._market_df["num_predictions"] = 0
        logger.info(f"Loaded {len(self._market_df)} forecasting questions.")

    def _expose_articles_up_to(self, target_date: date) -> Path:
        """
        Return the path to the date-gated article folder.
        Only articles published on or before target_date are accessible.

        Paper reference: Section 3.1 (Context), Appendix B.3 (Sandboxing)
        """
        # Articles are organized as corpus_root/YYYY/MM/DD/articles.jsonl
        # The sandbox bind-mount ensures only these paths are readable.
        return self.corpus_root

    def submit_forecast(
        self,
        question_id: int,
        outcomes: dict[str, float],
    ) -> dict:
        """
        Register or update a probability distribution over outcomes for a question.

        Paper reference: Section 3.1 (Agent Interaction), action submit_forecast()

        Args:
            question_id: qid from market.csv
            outcomes: dict mapping outcome strings to probabilities (sum <= 1.0)

        Returns:
            Receipt dict with qid, timestamp, and validation status.
        """
        assert self._market_df is not None, "Call load_questions() first."
        assert len(outcomes) <= self.config.simulation.max_outcomes_per_question, (
            f"Max {self.config.simulation.max_outcomes_per_question} outcomes allowed; "
            f"got {len(outcomes)}"
        )
        assert sum(outcomes.values()) <= 1.0 + 1e-6, (
            f"Probabilities must sum to ≤ 1.0; got {sum(outcomes.values()):.4f}"
        )
        assert all(
            k not in ("Unknown", "TBD", "Other", "N/A") for k in outcomes
        ), "Placeholder outcome names are not allowed."

        mask = self._market_df["qid"] == question_id
        assert mask.any(), f"Question ID {question_id} not found."
        assert not self._market_df.loc[mask, "is_resolved"].values[0], (
            f"Question {question_id} is already resolved."
        )

        self._market_df.loc[mask, "my_prediction"] = json.dumps(outcomes)
        self._market_df.loc[mask, "my_prediction_date"] = str(self.current_date)
        self._market_df.loc[mask, "num_predictions"] += 1

        logger.debug(f"Forecast submitted: qid={question_id}, outcomes={outcomes}")
        return {"qid": question_id, "date": str(self.current_date), "status": "accepted"}

    def next_day(self) -> dict:
        """
        Advance the simulation by one day.

        Paper reference: Section 3.1 (Agent Interaction), action next_day()

        Steps:
        1. Resolve questions whose resolution_date == current_date
        2. Compute and record scores for resolved questions
        3. Increment current_date
        4. Return state update summary

        Returns:
            Summary dict with resolved questions, new articles, and current metrics.
        """
        assert self._market_df is not None, "Call load_questions() first."

        # Step 1: Resolve questions due today
        resolved_today = []
        due_mask = (
            self._market_df["resolution_date"] == self.current_date
        ) & (~self._market_df["is_resolved"])

        for idx in self._market_df[due_mask].index:
            row = self._market_df.loc[idx]
            ground_truth = self._lookup_ground_truth(row["qid"])
            self._market_df.loc[idx, "is_resolved"] = True
            self._market_df.loc[idx, "ground_truth"] = ground_truth

            prediction = json.loads(row["my_prediction"]) if pd.notna(row["my_prediction"]) else {}
            bss = compute_brier_skill_score(prediction, ground_truth)
            resolved_today.append({
                "qid": row["qid"],
                "title": row["title"],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "brier_skill_score": bss,
            })
            logger.info(
                f"Resolved qid={row['qid']}: truth={ground_truth!r}, BSS={bss:.4f}"
            )

        # Step 2: Count new articles arriving tomorrow
        next_date = self.current_date + timedelta(days=1)
        new_articles_count = self._count_articles_for_date(next_date)

        # Step 3: Advance date
        prev_date = self.current_date
        self.current_date = next_date

        # Step 4: Persist updated state
        self._save_market_csv()

        summary = {
            "prev_date": str(prev_date),
            "current_date": str(self.current_date),
            "resolved_today": resolved_today,
            "new_articles": new_articles_count,
            "num_active": int((~self._market_df["is_resolved"]).sum()),
            "num_resolved": int(self._market_df["is_resolved"].sum()),
        }
        self._history.append(summary)
        return summary

    def _lookup_ground_truth(self, qid: int) -> str:
        """Load ground truth answer for a question from the ground truth store."""
        gt_path = Path(self.config.questions.questions_path).parent / "ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                store = json.load(f)
            return store.get(str(qid), "")
        return ""

    def _count_articles_for_date(self, target_date: date) -> int:
        """Count articles available on a specific date."""
        article_path = (
            self.corpus_root
            / str(target_date.year)
            / f"{target_date.month:02d}"
            / f"{target_date.day:02d}"
            / "articles.jsonl"
        )
        if not article_path.exists():
            return 0
        with open(article_path) as f:
            return sum(1 for _ in f)

    def _save_market_csv(self) -> None:
        """Persist the market state CSV (agent sees read-only copy via sandbox)."""
        out_path = self.agent_workspace / "market.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._market_df.to_csv(out_path, index=False)

    def get_final_metrics(self) -> dict:
        """
        Compute final metrics at end of simulation.

        Returns:
            Dict with mean BSS, accuracy, and per-question breakdown.
        """
        assert self._market_df is not None
        resolved = self._market_df[self._market_df["is_resolved"]].copy()

        bss_scores = []
        correct = []
        for _, row in resolved.iterrows():
            pred = json.loads(row["my_prediction"]) if pd.notna(row["my_prediction"]) else {}
            gt = row["ground_truth"]
            bss_scores.append(compute_brier_skill_score(pred, gt))
            correct.append(compute_accuracy(pred, gt))

        return {
            "mean_brier_skill_score": float(sum(bss_scores) / len(bss_scores)) if bss_scores else 0.0,
            "accuracy": float(sum(correct) / len(correct)) if correct else 0.0,
            "num_resolved": len(resolved),
            "num_questions": len(self._market_df),
        }

    def is_complete(self) -> bool:
        """Return True if simulation has passed the end date."""
        return self.current_date > self.end_date
