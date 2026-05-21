"""
Task Manager — CSV-based forecasting question and prediction state.

Paper reference: Section 3.1 (Tasks)
  "The current state of tasks is maintained as a CSV file, with each row containing data
   about one forecasting question ... question's background information, resolution criteria,
   resolution date, and the agent's most recent forecast."
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = [
    "qid", "title", "background", "resolution_criteria",
    "answer_type", "resolution_date", "is_resolved",
    "ground_truth", "my_prediction", "my_prediction_date", "num_predictions",
]


class TaskManager:
    """
    Manages the CSV state file for all forecasting questions.

    The CSV is the single source of truth for:
    - Question metadata (title, background, criteria, resolution date)
    - Agent's current predictions per question
    - Resolution status and ground truth (once resolved)

    Paper reference: Section 3.1 (Tasks), Appendix B.2 (Simulation Logic)
    """

    def __init__(self, questions_path: str, workspace_path: str):
        """
        Args:
            questions_path: Path to initial questions CSV (with ground truth)
            workspace_path: Agent workspace directory; market.csv written here (read-only for agent)
        """
        self.questions_path = Path(questions_path)
        self.workspace_path = Path(workspace_path)
        self._df: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        n = len(self._df) if self._df is not None else "not loaded"
        return f"TaskManager(questions={n}, workspace={self.workspace_path})"

    def load(self) -> None:
        """Load questions from CSV and initialise prediction columns."""
        assert self.questions_path.exists(), f"Questions not found: {self.questions_path}"
        self._df = pd.read_csv(self.questions_path)
        self._df["resolution_date"] = pd.to_datetime(self._df["resolution_date"]).dt.date

        # Initialise runtime columns if not present
        for col, default in [
            ("is_resolved", False),
            ("ground_truth", None),
            ("my_prediction", None),
            ("my_prediction_date", None),
            ("num_predictions", 0),
        ]:
            if col not in self._df.columns:
                self._df[col] = default

    def get_active(self, current_date) -> pd.DataFrame:
        """Return questions that are not yet resolved."""
        assert self._df is not None
        return self._df[~self._df["is_resolved"]].copy()

    def get_resolving_on(self, target_date) -> pd.DataFrame:
        """Return questions whose resolution_date == target_date."""
        assert self._df is not None
        return self._df[
            (self._df["resolution_date"] == target_date) & (~self._df["is_resolved"])
        ].copy()

    def set_prediction(self, qid: int, outcomes: dict, prediction_date: str) -> None:
        """Record an agent's prediction for a question."""
        assert self._df is not None
        mask = self._df["qid"] == qid
        assert mask.any(), f"qid={qid} not found"
        self._df.loc[mask, "my_prediction"] = json.dumps(outcomes)
        self._df.loc[mask, "my_prediction_date"] = prediction_date
        self._df.loc[mask, "num_predictions"] += 1

    def resolve(self, qid: int, ground_truth: str) -> None:
        """Mark a question as resolved with its ground truth answer."""
        assert self._df is not None
        mask = self._df["qid"] == qid
        self._df.loc[mask, "is_resolved"] = True
        self._df.loc[mask, "ground_truth"] = ground_truth

    def write_market_csv(self) -> Path:
        """
        Write the current state as market.csv to the agent workspace (read-only for agent).
        Returns the path written.
        """
        assert self._df is not None
        out = self.workspace_path / "market.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(out, index=False)
        return out

    def summary(self) -> dict:
        """Return a summary dict of current task state."""
        assert self._df is not None
        return {
            "total": len(self._df),
            "active": int((~self._df["is_resolved"]).sum()),
            "resolved": int(self._df["is_resolved"].sum()),
            "with_prediction": int(self._df["my_prediction"].notna().sum()),
        }
