"""
evolvemem/evolvemem.py

Top-level EvolveMem orchestrator combining all three layers.
Implements the complete pipeline from Algorithm 2 (Appendix B) of:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

Three layers (Figure 2):
  1. Structured Memory Store (Section 3.1)
  2. Multi-View Retrieval + Answer Generation (Section 3.2)
  3. Self-Evolution Engine (Section 3.3)
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Dict, Any, Optional

import yaml

from .memory.store import MemoryStore
from .memory.extractor import MemoryExtractor
from .memory.consolidator import Consolidator
from .retrieval.config import RetrievalConfig
from .retrieval.retriever import MultiViewRetriever
from .retrieval.answer_gen import AnswerGenerator
from .evolution.engine import EvolutionEngine
from .evolution.diagnosis import DiagnosisModule
from .embeddings.encoder import get_embedder
from .llm.client import LLMClient

logger = logging.getLogger(__name__)


class EvolveMem:
    """
    Top-level EVOLVEMEM orchestrator.

    Wires together all components and exposes the main interface:
      - ingest_sessions(): extract and store memories from conversations
      - answer(): answer a single query using current config
      - evolve(): run self-evolution loop to optimize retrieval config
      - save_state() / load_state(): persist/restore config

    Paper reference: Figure 2, Algorithm 2 (Appendix B)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize from YAML config.

        Args:
            config_path: Path to config.yaml (see configs/config.yaml for schema).
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        mem_cfg = cfg.get("memory", {})
        ret_cfg = cfg.get("retrieval_initial", {})
        evo_cfg = cfg.get("evolution", {})
        llm_cfg = cfg.get("llm", {})

        # LLM client
        self.llm_client = LLMClient(
            provider=llm_cfg.get("provider", "openai"),
            model=llm_cfg.get("answer_model", "gpt-4o"),
            max_tokens=llm_cfg.get("max_tokens", 2048),
        )

        # Embedder
        embedder = get_embedder(
            model_name=mem_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"),
        )

        # Layer 1: Memory store
        self.store = MemoryStore(
            db_path=mem_cfg.get("db_path", "evolvemem.db"),
            embedder=embedder,
        )
        self.extractor = MemoryExtractor(
            llm_client=self.llm_client,
            window_size=mem_cfg.get("window_size", 40),
            sub_window_size=mem_cfg.get("sub_window_size", 15),
            max_retries=mem_cfg.get("max_retries", 3),
        )
        self.consolidator = Consolidator(
            tau_j=mem_cfg.get("tau_j", 0.80),
            alpha_d=mem_cfg.get("alpha_d", 0.05),
            iota_min=mem_cfg.get("iota_min", 0.15),
            delta_rho=mem_cfg.get("delta_rho", 0.05),
            rho_max=mem_cfg.get("rho_max", 0.30),
        )

        # Layer 2: Retrieval
        self.config = RetrievalConfig.from_dict(ret_cfg) if ret_cfg else RetrievalConfig.initial()
        self.retriever = MultiViewRetriever(
            store=self.store,
            config=self.config,
            llm_client=self.llm_client,
            lambda_iota=llm_cfg.get("lambda_iota", 1.0),   # ASSUMED
            lambda_r=llm_cfg.get("lambda_r", 1.0),          # ASSUMED
            rrf_k=evo_cfg.get("rrf_k", 60),                 # ASSUMED
        )
        self.answer_gen = AnswerGenerator(
            llm_client=self.llm_client,
            tau_ver=llm_cfg.get("tau_ver", 0.5),            # ASSUMED
        )

        # Layer 3: Evolution
        self.diagnosis = DiagnosisModule(llm_client=self.llm_client)
        self.evo_cfg = evo_cfg
        self._conversations: List[List[dict]] = []

        os.makedirs(cfg.get("paths", {}).get("log_dir", "logs/"), exist_ok=True)
        self._log_dir = cfg.get("paths", {}).get("log_dir", "logs/")

        logger.info("EvolveMem initialized.")

    # ------------------------------------------------------------------
    # Ingestion (Layer 1)
    # ------------------------------------------------------------------

    def ingest_sessions(self, sessions: List[List[dict]]) -> None:
        """
        Extract and store memories from conversation sessions.

        Implements the Ingestion Phase from Algorithm 2 (Appendix B):
          - Extract memory units via sliding window
          - Validate and pre-deduplicate
          - Generate embeddings
          - Add to store
          - Run consolidation pass

        Args:
            sessions: List of sessions; each session is a list of turn dicts
                      with "speaker" and "text" keys.
        """
        self._conversations = sessions  # Keep for targeted re-extraction

        for i, session in enumerate(sessions):
            logger.info(f"Ingesting session {i+1}/{len(sessions)} ({len(session)} turns)")
            existing = self.store.get_all()
            units = self.extractor.extract(session, existing_memories=existing)
            self.store.add(units)
            logger.info(f"  → Added {len(units)} units from session {i+1}")

        # Consolidation pass
        self.consolidator.run_all(self.store)
        logger.info(f"Ingestion complete. Store size: {self.store.size()} units")

    # ------------------------------------------------------------------
    # Query answering (Layer 2)
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        category: Optional[int] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> str:
        """
        Answer a single query using the current (or provided) retrieval config.

        Runs entity reinforcement consolidation after retrieval.

        Args:
            query: Query string.
            category: Question category for per-category overrides (1–5 for LoCoMo).
            config: Override config (defaults to self.config).

        Returns:
            Answer string.
        """
        cfg = config or self.config
        retrieved = self.retriever.retrieve(query, config=cfg, category=category)
        self.consolidator.reinforce_entities(retrieved, query)
        return self.answer_gen.generate(query, retrieved, config=cfg, category=category)

    # ------------------------------------------------------------------
    # Self-evolution (Layer 3)
    # ------------------------------------------------------------------

    def evolve(
        self,
        qa_pairs: List[Dict[str, Any]],
        update_config: bool = True,
    ) -> RetrievalConfig:
        """
        Run the self-evolution loop to optimize retrieval configuration.

        "Starting from a minimal baseline, the process converges autonomously."
        — Abstract

        Args:
            qa_pairs: Evaluation set: [{"q": ..., "ref": ..., "category": ...}, ...]
            update_config: If True, update self.config with the best found config.

        Returns:
            Best RetrievalConfig theta* found during evolution.
        """
        flat_conversations = []
        for sess in self._conversations:
            flat_conversations.extend(sess)

        engine = EvolutionEngine(
            store=self.store,
            retriever=self.retriever,
            answer_gen=self.answer_gen,
            diagnosis=self.diagnosis,
            extractor=self.extractor,
            conversations=flat_conversations,
            log_dir=self._log_dir,
        )

        best_config = engine.run(
            qa_pairs=qa_pairs,
            initial_config=self.config,
            max_rounds=self.evo_cfg.get("max_rounds", 7),
            epsilon=self.evo_cfg.get("epsilon", 0.005),
            tau_rev=self.evo_cfg.get("tau_rev", 0.01),
        )

        if update_config:
            self.config = best_config
            self.retriever.config = best_config
            logger.info(f"Updated self.config to best evolved config: {best_config}")

        return best_config

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save current retrieval configuration to JSON."""
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")

    def load_state(self, path: str) -> None:
        """Load retrieval configuration from JSON."""
        with open(path, "r") as f:
            d = json.load(f)
        self.config = RetrievalConfig.from_dict(d)
        self.retriever.config = self.config
        logger.info(f"Config loaded from {path}: {self.config}")
