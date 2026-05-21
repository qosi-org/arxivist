"""
Hybrid News Corpus Retrieval
=============================
Implements the hybrid semantic + keyword search over CCNews.

Paper reference: Section 4.1 (Search Corpus):
  "We provide access to a hybrid semantic + keyword search tool over the news corpus,
   implemented using LanceDB, that returns 5 chunks of 512 tokens.
   We use Qwen3 8B embeddings for the semantic search."
   search_news(query, from_date, to_date)

The retrieval system enforces date-gating to prevent future information leakage
(Appendix B.3, B.5).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

try:
    import lancedb
    import pyarrow as pa
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class NewsRetriever:
    """
    Hybrid retrieval over a LanceDB-indexed CCNews corpus.

    Provides date-range-controlled semantic + keyword search.
    Returns up to `chunks_per_query` chunks of `chunk_size` tokens.

    Paper reference: Section 4.1, Appendix B.5
    """

    def __init__(
        self,
        index_path: str,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-8B",
        chunks_per_query: int = 5,
        chunk_size: int = 512,
    ):
        """
        Args:
            index_path: Path to LanceDB database directory
            embedding_model_name: HuggingFace model ID for semantic embeddings
            chunks_per_query: Number of chunks to return per search (paper: 5)
            chunk_size: Token size of each chunk (paper: 512)
        """
        assert chunks_per_query > 0, "chunks_per_query must be positive"
        assert chunk_size > 0, "chunk_size must be positive"

        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model_name
        self.chunks_per_query = chunks_per_query
        self.chunk_size = chunk_size

        self._db = None
        self._table = None
        self._embedder = None
        self._current_date_cap: Optional[date] = None

    def __repr__(self) -> str:
        return (
            f"NewsRetriever(index={self.index_path}, "
            f"model={self.embedding_model_name}, "
            f"date_cap={self._current_date_cap})"
        )

    def connect(self) -> None:
        """Connect to the LanceDB index and load the embedding model."""
        if not HAS_LANCEDB:
            raise ImportError("lancedb is required: pip install lancedb")
        if not HAS_ST:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")

        assert self.index_path.exists(), f"LanceDB index not found: {self.index_path}"
        self._db = lancedb.connect(str(self.index_path))
        self._table = self._db.open_table("news_chunks")

        self._embedder = SentenceTransformer(self.embedding_model_name)

    def set_date_cap(self, cap: date) -> None:
        """
        Set the maximum date for search results.
        Prevents future information leakage (Appendix B.3).

        Args:
            cap: Maximum publication date allowed in results
        """
        self._current_date_cap = cap

    def search(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid semantic + keyword search over the date-gated news corpus.

        Paper reference: Section 4.1
          "search_news(query, from_date, to_date)"
          Returns 5 chunks of 512 tokens.

        Args:
            query: Natural language search query
            from_date: ISO date string (inclusive lower bound), e.g. "2026-01-01"
            to_date: ISO date string (inclusive upper bound); capped at current_date_cap

        Returns:
            List of up to chunks_per_query dicts with keys:
              {text, source, url, pub_date, score}
        """
        assert self._table is not None, "Call connect() before searching."
        assert query.strip(), "Query must be non-empty."

        # Enforce date cap — critical for leakage prevention (Appendix B.3)
        effective_to = to_date
        if self._current_date_cap is not None:
            cap_str = str(self._current_date_cap)
            if effective_to is None or effective_to > cap_str:
                effective_to = cap_str

        # Build date filter
        filters = []
        if from_date:
            filters.append(f"pub_date >= '{from_date}'")
        if effective_to:
            filters.append(f"pub_date <= '{effective_to}'")
        where_clause = " AND ".join(filters) if filters else None

        # Semantic embedding
        query_embedding = self._embedder.encode(query, normalize_embeddings=True).tolist()

        # Hybrid search via LanceDB
        # WARNING: low-confidence — hybrid fusion method not specified in paper (Appendix B.5)
        # TODO: verify fusion weights from paper authors
        search_builder = (
            self._table.search(query_embedding, query_type="hybrid")
            .limit(self.chunks_per_query)
        )
        if where_clause:
            search_builder = search_builder.where(where_clause)

        results = search_builder.to_list()
        return [
            {
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "url": r.get("url", ""),
                "pub_date": r.get("pub_date", ""),
                "score": r.get("_relevance_score", 0.0),
            }
            for r in results
        ]


class CCNewsIndexBuilder:
    """
    Builds a LanceDB hybrid index from raw CCNews JSONL files.

    Paper reference: Section 4.1, Appendix B.5
    CCNews JSONL format: {"text": "...", "url": "...", "pub_date": "YYYY-MM-DD", "source": "..."}
    Articles stored as: corpus_root/YYYY/MM/DD/articles.jsonl
    """

    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-8B",
        chunk_size: int = 512,
    ):
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size

    def __repr__(self) -> str:
        return f"CCNewsIndexBuilder(corpus={self.corpus_path}, index={self.index_path})"

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks of approximately chunk_size tokens.
        Uses whitespace tokenization as approximation (1 token ≈ 0.75 words).
        """
        words = text.split()
        approx_words_per_chunk = int(self.chunk_size * 0.75)
        chunks = []
        for i in range(0, len(words), approx_words_per_chunk):
            chunks.append(" ".join(words[i : i + approx_words_per_chunk]))
        return chunks

    def build(self, force: bool = False) -> None:
        """
        Ingest all CCNews JSONL files and build the LanceDB hybrid index.

        Args:
            force: Rebuild index even if it already exists.
        """
        if not HAS_LANCEDB:
            raise ImportError("lancedb is required: pip install lancedb")

        if self.index_path.exists() and not force:
            print(f"Index already exists at {self.index_path}. Use force=True to rebuild.")
            return

        if not HAS_ST:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")

        embedder = SentenceTransformer(self.embedding_model_name)
        db = lancedb.connect(str(self.index_path))

        records = []
        total_articles = 0
        for jsonl_path in sorted(self.corpus_path.rglob("articles.jsonl")):
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = article.get("text", "")
                    if not text.strip():
                        continue
                    chunks = self._chunk_text(text)
                    for chunk in chunks:
                        embedding = embedder.encode(chunk, normalize_embeddings=True).tolist()
                        records.append({
                            "text": chunk,
                            "url": article.get("url", ""),
                            "source": article.get("source", ""),
                            "pub_date": article.get("pub_date", ""),
                            "vector": embedding,
                        })
                    total_articles += 1

        print(f"Indexed {total_articles} articles → {len(records)} chunks")
        schema = pa.schema([
            pa.field("text", pa.string()),
            pa.field("url", pa.string()),
            pa.field("source", pa.string()),
            pa.field("pub_date", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
        ])
        table = db.create_table("news_chunks", data=records, schema=schema, mode="overwrite")
        # Enable full-text search for hybrid retrieval
        table.create_fts_index("text")
        print(f"Index written to {self.index_path}")
