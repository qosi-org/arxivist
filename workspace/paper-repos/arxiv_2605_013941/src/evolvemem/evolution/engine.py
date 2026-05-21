"""
evolvemem/evolution/engine.py

Self-evolution engine implementing Algorithm 1 from:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

The four-step loop: EVALUATE → DIAGNOSE → PROPOSE → GUARD
with automatic revert-on-regression and explore-on-stagnation.

Objective (Equation 3):
  theta* = argmax_{theta in Theta} F(theta; K, Q)
         = (1/|Q|) * sum score(yhat(q; theta, K), y*)

Update rule (Equation 4):
  theta_{r+1} = theta*_{r-1}           if f_{r-1} - f_r > tau_rev   [REVERT]
              = theta_r ⊕ eta_exp      if |f_r - f_{r-1}| < eps×2r  [EXPLORE]
              = clamp(theta_r ⊕ Delta) otherwise                      [APPLY]
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from ..memory.store import MemoryStore
from ..retrieval.config import RetrievalConfig
from ..retrieval.retriever import MultiViewRetriever
from ..retrieval.answer_gen import AnswerGenerator
from ..evaluation.metrics import Evaluator
from .diagnosis import DiagnosisModule

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """
    Self-evolution engine — implements Algorithm 1 (EVOLVEMEM Self-Evolution Loop).

    "Each evolution round constitutes an autonomous research iteration that is
    empirically validated before acceptance, realizing an AutoResearch process
    within the system itself." — Section 3.3

    Paper reference: Section 3.3, Algorithm 1
    """

    def __init__(
        self,
        store: MemoryStore,
        retriever: MultiViewRetriever,
        answer_gen: AnswerGenerator,
        diagnosis: DiagnosisModule,
        extractor=None,          # Optional for targeted re-extraction
        conversations=None,      # Source sessions for targeted re-extraction
        log_dir: str = "logs/",
    ):
        self.store = store
        self.retriever = retriever
        self.answer_gen = answer_gen
        self.diagnosis = diagnosis
        self.extractor = extractor
        self.conversations = conversations
        self.log_dir = log_dir
        self.evaluator = Evaluator()
        os.makedirs(log_dir, exist_ok=True)

    def run(
        self,
        qa_pairs: List[Dict[str, Any]],
        initial_config: Optional[RetrievalConfig] = None,
        max_rounds: int = 7,          # Section 4.1: Rmax=7
        epsilon: float = 0.005,       # Appendix A: convergence threshold
        tau_rev: float = 0.01,        # Algorithm 1: revert threshold
    ) -> RetrievalConfig:
        """
        Run the self-evolution loop (Algorithm 1).

        "Starting from a minimal baseline, the process converges autonomously,
        discovering effective retrieval strategies including entirely new
        configuration dimensions not present in the original action space."
        — Abstract

        Args:
            qa_pairs: List of {"q": ..., "ref": ..., "category": ...} dicts.
            initial_config: Starting theta_0 (defaults to RetrievalConfig.initial()).
            max_rounds: Maximum evolution rounds Rmax (default 7, Section 4.1).
            epsilon: Convergence threshold (default 0.005 = 0.5pp, Appendix A).
            tau_rev: Revert-on-regression threshold (default 0.01, Algorithm 1).

        Returns:
            Best configuration theta* found during evolution.
        """
        theta = initial_config or RetrievalConfig.initial()
        f_star = 0.0
        theta_star = deepcopy(theta)
        f_prev = 0.0
        stagnation_count = 0

        logger.info(f"Starting evolution loop: max_rounds={max_rounds}, epsilon={epsilon}, tau_rev={tau_rev}")
        logger.info(f"Initial config: {theta}")

        for r in range(max_rounds + 1):  # R0 through Rmax
            logger.info(f"\n{'='*60}\nEvolution Round R{r}\n{'='*60}")

            # EVALUATE — Algorithm 1 lines 4-5
            f_r, raw_log = self._evaluate(qa_pairs, theta)
            logger.info(f"R{r}: F1={f_r:.4f} (prev={f_prev:.4f}, best={f_star:.4f})")

            # Write per-question raw log (Equation 14)
            self._write_raw_log(raw_log, round_num=r)

            # Update best
            if f_r > f_star:
                f_star = f_r
                theta_star = deepcopy(theta)
                logger.info(f"  → New best: {f_star:.4f}")

            # Convergence check (Appendix A Equation 16)
            if r > 0 and (f_r - f_prev) < epsilon:
                logger.info(f"  → Converged (improvement {f_r - f_prev:.4f} < epsilon {epsilon}). Stopping.")
                break

            if r == max_rounds:
                logger.info(f"  → Reached max_rounds={max_rounds}. Stopping.")
                break

            # DIAGNOSE — Algorithm 1 line 6
            proposal = self.diagnosis.diagnose(
                raw_log=raw_log,
                current_config=theta,
                memory_size=self.store.size(),
            )
            delta = proposal.get("parameter_suggestions", {})
            missing_topics = proposal.get("missing_topics", [])

            # Coverage-gap triggered re-extraction (Algorithm 1 lines 13-14, Equation 15)
            if missing_topics and self.extractor and self.conversations:
                logger.info(f"  → Coverage gap detected: {missing_topics}. Triggering targeted re-extraction.")
                new_units = self.extractor.extract_targeted(self.conversations, missing_topics)
                if new_units:
                    self.store.add(new_units)
                    logger.info(f"  → Added {len(new_units)} new units from targeted re-extraction.")

            # GUARD — Algorithm 1 lines 7-12 (Equation 4)
            theta, stagnation_count = self._update(
                theta_r=theta,
                delta=delta,
                f_r=f_r,
                f_prev=f_prev,
                f_star=f_star,
                theta_star=theta_star,
                tau_rev=tau_rev,
                epsilon=epsilon,
                stagnation_count=stagnation_count,
                round_num=r,
            )

            f_prev = f_r

        logger.info(f"\nEvolution complete. Best F1={f_star:.4f} with config: {theta_star}")
        return theta_star

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        qa_pairs: List[Dict[str, Any]],
        config: RetrievalConfig,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate config over all QA pairs and produce per-question raw log L_r.

        Raw log format (Equation 14):
          L_r = {(q_j, yhat_j, y*_j, score_j, R(q_j; theta_r))}_{j=1}^{|Q|}

        Paper reference: Algorithm 1 lines 4-5, Appendix A Equation 14
        """
        raw_log = []
        total_score = 0.0

        for item in qa_pairs:
            query = item.get("q", item.get("question", ""))
            reference = item.get("ref", item.get("reference", item.get("answer", "")))
            category = item.get("category")

            # Retrieve
            cfg_for_cat = config.for_category(category)
            retrieved = self.retriever.retrieve(query, config=cfg_for_cat, category=category)

            # Generate answer
            prediction = self.answer_gen.generate(query, retrieved, config=config, category=category)

            # Score
            score = self.evaluator.token_f1(prediction, reference)
            total_score += score

            raw_log.append({
                "q": query,
                "pred": prediction,
                "ref": reference,
                "score": score,
                "category": category,
                "sources": [u.memory_id for u in retrieved[:5]],  # top-5 source IDs
            })

        overall_f1 = total_score / max(len(qa_pairs), 1)
        return overall_f1, raw_log

    # ------------------------------------------------------------------
    # Update rule (Equation 4)
    # ------------------------------------------------------------------

    def _update(
        self,
        theta_r: RetrievalConfig,
        delta: Dict[str, Any],
        f_r: float,
        f_prev: float,
        f_star: float,
        theta_star: RetrievalConfig,
        tau_rev: float,
        epsilon: float,
        stagnation_count: int,
        round_num: int,
    ) -> Tuple[RetrievalConfig, int]:
        """
        Apply three-branch update rule from Algorithm 1 / Equation 4.

        Branch 1 — REVERT: if f_{r-1} - f_r > tau_rev → revert to best-so-far
        Branch 2 — EXPLORE: if stagnation for 2 rounds → random perturbation
        Branch 3 — APPLY: apply diagnosis-proposed adjustment (normal case)
        """
        regression = f_prev - f_r if round_num > 0 else 0.0
        improvement = abs(f_r - f_prev)

        # Branch 1: REVERT (Algorithm 1 lines 7-8)
        if round_num > 0 and regression > tau_rev:
            logger.info(f"  → REVERT: regression {regression:.4f} > tau_rev {tau_rev}. Reverting to best.")
            return deepcopy(theta_star), 0

        # Branch 2: EXPLORE (Algorithm 1 lines 9-10)
        if round_num > 0 and improvement < epsilon:
            stagnation_count += 1
        else:
            stagnation_count = 0

        if stagnation_count >= 2:
            logger.info(f"  → EXPLORE: stagnation for {stagnation_count} rounds. Applying random perturbation.")
            return self._random_perturbation(theta_r), 0

        # Branch 3: APPLY (Algorithm 1 lines 11-12)
        if delta:
            new_config = theta_r.apply_delta(delta)
            logger.info(f"  → APPLY: proposed {len(delta)} parameter changes → {new_config}")
            return new_config, stagnation_count
        else:
            logger.info(f"  → No delta proposed. Config unchanged.")
            return deepcopy(theta_r), stagnation_count

    @staticmethod
    def _random_perturbation(config: RetrievalConfig) -> RetrievalConfig:
        """
        Apply random perturbation to escape local optima (eta_exp in Equation 4).

        Paper reference: Section 3.3 "Update rule" — "random perturbation sampled
        to escape local optima"
        """
        cfg = deepcopy(config)

        # Randomly perturb a subset of parameters
        perturbations = [
            ("keyword_top_k", random.randint(-2, 3)),
            ("semantic_top_k", random.randint(-2, 3)),
            ("max_context", random.randint(-2, 3)),
            ("w_sem", random.uniform(-0.2, 0.2)),
            ("w_kw", random.uniform(-0.2, 0.2)),
        ]

        chosen = random.sample(perturbations, k=min(3, len(perturbations)))
        for attr, delta in chosen:
            current = getattr(cfg, attr, 0)
            setattr(cfg, attr, current + delta)

        return cfg.clamp()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_raw_log(self, raw_log: List[Dict[str, Any]], round_num: int) -> None:
        """
        Write per-question raw log to disk.

        "After each evaluation round r, the system writes a per-question raw log
        containing every question, prediction, ground-truth answer, score, and
        retrieved sources." — Section 3.3, Equation 14
        """
        path = os.path.join(self.log_dir, f"raw_results_r{round_num}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for record in raw_log:
                f.write(json.dumps(record) + "\n")
        logger.debug(f"Wrote raw log to {path}")
