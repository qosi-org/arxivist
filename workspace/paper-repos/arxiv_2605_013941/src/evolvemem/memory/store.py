"""
evolvemem/memory/store.py

SQLite/FTS5-backed typed memory store.
Implements Layer 1 (Structured Memory Store) from Section 3.1 of:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

SQLite schema described in Appendix D.1.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import numpy as np


# Six-category memory taxonomy (Section 3.1)
MEMORY_TYPES = {
    "episodic",
    "semantic",
    "preference",
    "project_state",
    "working_summary",
    "procedural",
}


@dataclass
class MemoryUnit:
    """
    A single memory unit — the atomic storage element of EVOLVEMEM.

    Represents m = (c, μ, e, η) as defined in Section 3.1:
      c   : natural-language content string
      mu  : memory type from six-category taxonomy T
      e   : dense embedding vector (stored as BLOB in SQLite)
      eta : auxiliary metadata dict

    Paper reference: Section 3.1 "Memory representation"
    """

    content: str                              # c: natural-language content
    memory_type: str = "episodic"             # μ ∈ T (six-category taxonomy)
    embedding: Optional[np.ndarray] = None    # e ∈ R^d (d=768 for BGE)
    importance: float = 0.5                   # ι_i ∈ [iota_min, 1.0]
    confidence: float = 0.8
    reinforcement_score: float = 0.0          # ρ_i, updated by entity co-occurrence
    entities: List[str] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None           # ISO format date string
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope_id: str = "user:default|workspace:default|session:default"

    def __post_init__(self):
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(
                f"memory_type must be one of {MEMORY_TYPES}, got '{self.memory_type}'"
            )

    def __repr__(self) -> str:
        return (
            f"MemoryUnit(id={self.memory_id[:8]}, type={self.memory_type}, "
            f"importance={self.importance:.2f}, content='{self.content[:60]}...')"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (embedding excluded — stored separately as BLOB)."""
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> "MemoryUnit":
        """Deserialize from dict, optionally attaching a pre-loaded embedding."""
        d = dict(d)
        d.pop("embedding", None)
        unit = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        unit.embedding = embedding
        return unit


_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memories (
    memory_id       TEXT PRIMARY KEY,
    scope_id        TEXT NOT NULL DEFAULT 'user:default|workspace:default|session:default',
    memory_type     TEXT NOT NULL,
    content         TEXT NOT NULL,
    entities        TEXT NOT NULL DEFAULT '[]',
    persons         TEXT NOT NULL DEFAULT '[]',
    locations       TEXT NOT NULL DEFAULT '[]',
    topics          TEXT NOT NULL DEFAULT '[]',
    keywords        TEXT NOT NULL DEFAULT '[]',
    importance      REAL NOT NULL DEFAULT 0.5,
    confidence      REAL NOT NULL DEFAULT 0.8,
    reinforcement_score REAL NOT NULL DEFAULT 0.0,
    embedding       BLOB,
    timestamp       TEXT,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    topics,
    entities,
    content='memories',
    content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO schema_version VALUES (6);
"""


class MemoryStore:
    """
    SQLite/FTS5-backed typed memory store for EVOLVEMEM.

    Implements the storage layer described in Section 3.1 and Appendix D.1.
    Supports BM25 (via FTS5), semantic (cosine), and structured (entity) retrieval views.

    Paper reference: Section 3.1, Appendix D.1
    """

    def __init__(self, db_path: str, embedder=None):
        """
        Args:
            db_path: Path to SQLite database file (use ':memory:' for in-memory).
            embedder: Embedding backend instance (SentenceTransformerEmbedder or HashingEmbedder).
                      If None, semantic search is disabled.
        """
        self.db_path = db_path
        self.embedder = embedder
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, units: List[MemoryUnit]) -> None:
        """
        Insert memory units into the store.
        Computes embeddings if embedder is available and unit has none.

        Paper reference: Section 3.1, Memory extraction output
        """
        if not units:
            return

        # Batch-encode embeddings
        if self.embedder is not None:
            texts_to_embed = [u for u in units if u.embedding is None]
            if texts_to_embed:
                vecs = self.embedder.encode([u.content for u in texts_to_embed])
                for unit, vec in zip(texts_to_embed, vecs):
                    unit.embedding = vec

        with self._conn:
            for unit in units:
                emb_blob = unit.embedding.astype(np.float32).tobytes() if unit.embedding is not None else None
                self._conn.execute(
                    """INSERT OR REPLACE INTO memories
                       (memory_id, scope_id, memory_type, content, entities, persons,
                        locations, topics, keywords, importance, confidence,
                        reinforcement_score, embedding, timestamp, created_at, status)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        unit.memory_id, unit.scope_id, unit.memory_type, unit.content,
                        json.dumps(unit.entities), json.dumps(unit.persons),
                        json.dumps(unit.locations), json.dumps(unit.topics),
                        json.dumps(unit.keywords), unit.importance, unit.confidence,
                        unit.reinforcement_score, emb_blob,
                        unit.timestamp, unit.created_at, "active",
                    ),
                )
                # Keep FTS5 index in sync
                self._conn.execute(
                    "INSERT OR REPLACE INTO memories_fts(memory_id, content, topics, entities) VALUES (?,?,?,?)",
                    (unit.memory_id, unit.content, json.dumps(unit.topics), json.dumps(unit.entities)),
                )

    # ------------------------------------------------------------------
    # BM25 / FTS5 retrieval (lexical view)
    # ------------------------------------------------------------------

    def search_bm25(self, query: str, top_k: int) -> List[Tuple[MemoryUnit, float]]:
        """
        Lexical search using SQLite FTS5 (BM25).

        Implements s_kw(q, m_i) = BM25(q, c_i) from Section 3.2 / Equation 2.
        SQLite FTS5 uses BM25 with k1=1.2, b=0.75 internally (approximate).

        Returns list of (MemoryUnit, bm25_score) sorted by score descending.
        Note: FTS5 rank() returns negative values; we negate for ascending convention.
        """
        if top_k <= 0 or not query.strip():
            return []

        rows = self._conn.execute(
            """SELECT m.*, -fts.rank AS bm25_score
               FROM memories_fts fts
               JOIN memories m ON fts.memory_id = m.memory_id
               WHERE memories_fts MATCH ?
               ORDER BY fts.rank
               LIMIT ?""",
            (self._sanitize_fts_query(query), top_k),
        ).fetchall()

        return [(self._row_to_unit(r), float(r["bm25_score"])) for r in rows]

    # ------------------------------------------------------------------
    # Semantic retrieval (cosine similarity)
    # ------------------------------------------------------------------

    def search_semantic(self, query: str, top_k: int, query_vec: Optional[np.ndarray] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        Semantic retrieval via cosine similarity.

        Implements s_sem(q, m_i) = cos(e_q, e_i) from Section 3.2 / Equation 3.

        Args:
            query: Query string (used to compute query_vec if not provided).
            top_k: Number of candidates.
            query_vec: Pre-computed query embedding (optional, avoids double encode).

        Returns list of (MemoryUnit, cosine_score) sorted by score descending.
        """
        if top_k <= 0 or self.embedder is None:
            return []

        if query_vec is None:
            query_vec = self.embedder.encode([query])[0]

        # Load all embeddings (feasible for memory stores of ~1000 units)
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE status='active' AND embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        embeddings = np.stack([
            np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
        ])

        # Eq. 3: cos(e_q, e_i)
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        e_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        scores = e_norms @ q_norm  # [N]

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._row_to_unit(rows[i]), float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Structured metadata retrieval
    # ------------------------------------------------------------------

    def search_structured(self, query: str, top_k: int, query_entities: Optional[List[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        Entity/location/person-based structured retrieval.

        Implements s_str(q, m_i) = sum_f I[extract_f(q) ∩ η_{i,f} ≠ ∅]
        from Section 3.2 / Equation 4.

        Args:
            query: Query string (entities extracted inline if query_entities is None).
            top_k: Number of candidates.
            query_entities: Pre-extracted entities from query (optional).

        Returns list of (MemoryUnit, structured_score) sorted by score descending.
        """
        if top_k <= 0:
            return []

        if query_entities is None:
            # Simple noun-phrase extraction via capitalized words (approximation)
            import re
            query_entities = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", query)

        if not query_entities:
            return []

        rows = self._conn.execute(
            "SELECT * FROM memories WHERE status='active'"
        ).fetchall()

        scored = []
        for row in rows:
            persons = set(json.loads(row["persons"] or "[]"))
            locations = set(json.loads(row["locations"] or "[]"))
            entities = set(json.loads(row["entities"] or "[]"))
            q_set = set(query_entities)

            # Eq. 4: count matching field types
            score = float(
                bool(q_set & persons) +
                bool(q_set & locations) +
                bool(q_set & entities)
            )
            if score > 0:
                scored.append((self._row_to_unit(row), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_all(self, status: str = "active") -> List[MemoryUnit]:
        """Return all memory units with given status."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE status=?", (status,)
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def size(self) -> int:
        """Return count of active memory units."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]

    def update_unit(self, memory_id: str, **kwargs) -> None:
        """Update specific fields of a memory unit."""
        allowed = {"importance", "reinforcement_score", "confidence", "status"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k}=?" for k in updates)
        self._conn.execute(
            f"UPDATE memories SET {set_clause} WHERE memory_id=?",
            (*updates.values(), memory_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_unit(self, row: sqlite3.Row) -> MemoryUnit:
        """Convert a SQLite row to a MemoryUnit."""
        emb = None
        if row["embedding"] is not None:
            emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        return MemoryUnit(
            memory_id=row["memory_id"],
            scope_id=row["scope_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            entities=json.loads(row["entities"] or "[]"),
            persons=json.loads(row["persons"] or "[]"),
            locations=json.loads(row["locations"] or "[]"),
            topics=json.loads(row["topics"] or "[]"),
            keywords=json.loads(row["keywords"] or "[]"),
            importance=row["importance"],
            confidence=row["confidence"],
            reinforcement_score=row["reinforcement_score"],
            embedding=emb,
            timestamp=row["timestamp"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Escape FTS5 special characters."""
        # Replace problematic chars; wrap in quotes for phrase-like matching
        cleaned = query.replace('"', '""').replace("'", "''")
        return f'"{cleaned}"'
