"""
evolvemem/evaluation/metrics.py

Token-level F1 and BLEU-1 scoring for EVOLVEMEM evaluation.
These are the primary metrics used on LoCoMo (Section 4.1 / Table 2).
MemBench uses exact-match accuracy (Table 3).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Dict


class Evaluator:
    """
    Evaluation metrics for EVOLVEMEM benchmarks.

    - LoCoMo: token-level F1 and BLEU-1 (Table 2)
    - MemBench: exact-match accuracy (Table 3)

    Paper reference: Section 4.1 "Protocols & Baselines"
    """

    def token_f1(self, prediction: str, reference: str) -> float:
        """
        Compute token-level F1 between prediction and reference.

        Standard QA token-F1: precision = |pred ∩ ref| / |pred|,
        recall = |pred ∩ ref| / |ref|, F1 = 2 * P * R / (P + R).

        Paper reference: Section 4.1 — primary LoCoMo metric
        """
        pred_tokens = self._normalize_and_tokenize(prediction)
        ref_tokens = self._normalize_and_tokenize(reference)

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)

        common = sum((pred_counts & ref_counts).values())
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def bleu1(self, prediction: str, reference: str) -> float:
        """
        Compute BLEU-1 (unigram precision with brevity penalty).

        Paper reference: Section 4.1 — secondary LoCoMo metric (Table 2)
        """
        pred_tokens = self._normalize_and_tokenize(prediction)
        ref_tokens = self._normalize_and_tokenize(reference)

        if not pred_tokens:
            return 0.0
        if not ref_tokens:
            return 0.0

        ref_counts = Counter(ref_tokens)
        clipped_matches = sum(
            min(count, ref_counts[token])
            for token, count in Counter(pred_tokens).items()
        )

        precision = clipped_matches / len(pred_tokens)

        # Brevity penalty
        bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else (len(pred_tokens) / len(ref_tokens))

        return bp * precision

    def exact_match(self, prediction: str, reference: str) -> float:
        """
        Exact-match accuracy (used for MemBench multiple-choice, Table 3).

        Returns 1.0 if normalized prediction matches reference, 0.0 otherwise.
        """
        return float(
            self._normalize(prediction) == self._normalize(reference)
        )

    def score_qa_set(
        self,
        predictions: List[str],
        references: List[str],
        mode: str = "f1",
    ) -> Dict[str, float]:
        """
        Score a full set of QA predictions.

        Args:
            predictions: List of predicted answer strings.
            references: List of reference answer strings.
            mode: "f1" for token-F1, "em" for exact match.

        Returns:
            Dict with overall score and count.
        """
        assert len(predictions) == len(references), "Prediction/reference count mismatch"

        if mode == "em":
            scores = [self.exact_match(p, r) for p, r in zip(predictions, references)]
        else:
            scores = [self.token_f1(p, r) for p, r in zip(predictions, references)]

        return {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "count": len(scores),
            "zero_count": sum(1 for s in scores if s == 0.0),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip punctuation and articles."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    def _normalize_and_tokenize(self, text: str) -> List[str]:
        return self._normalize(text).split()
