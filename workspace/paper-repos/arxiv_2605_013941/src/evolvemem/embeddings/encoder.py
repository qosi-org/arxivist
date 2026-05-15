"""
evolvemem/embeddings/encoder.py

Embedding backends for EVOLVEMEM semantic retrieval.
Implements Appendix D.2 from:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

Two backends (Appendix D.2):
  - SentenceTransformerEmbedder: BAAI/bge-base-en-v1.5, 768-dim, batch_size=32
  - HashingEmbedder: SHA-256 hash-based, 64-dim, zero external dependencies
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class EmbedderBase(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            np.ndarray of shape [len(texts), d] with L2-normalized embeddings.
        """
        ...


class SentenceTransformerEmbedder(EmbedderBase):
    """
    BGE-base embedding model (768-dim) using sentence-transformers.

    "Uses BAAI/bge-base-en-v1.5 (768-dim) from the sentence-transformers library.
    Provides semantic similarity for hybrid retrieval. Batch encoding with size 32
    for efficiency." — Appendix D.2

    All experiments use this embedder (Appendix D.2).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 32,        # Appendix D.2: batch_size=32
        device: str = "cpu",
    ):
        """
        Args:
            model_name: HuggingFace model identifier (default: BGE-base).
            batch_size: Encoding batch size (Appendix D.2: 32).
            device: Compute device ("cpu" or "cuda").
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dim = 768
        logger.info(f"Loaded SentenceTransformer: {model_name} (dim={self.dim})")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using BGE model with L2 normalization.

        Returns np.ndarray of shape [N, 768] with unit-norm vectors.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # L2 norm for cosine similarity
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder(dim={self.dim}, batch_size={self.batch_size})"


class HashingEmbedder(EmbedderBase):
    """
    Lightweight SHA-256 hash-based embedder (64-dim, zero ML dependencies).

    "A lightweight, deterministic hash-based embedder that maps tokens to dimensions
    via SHA-256 hashing. Produces d=64 dimensional vectors with ℓ2 normalization.
    Zero external dependencies; suitable for environments where installing ML libraries
    is impractical." — Appendix D.2

    NOTE: Semantic similarity quality is much lower than the BGE embedder.
    Use only as a fallback for testing or dependency-free environments.
    """

    def __init__(self, dim: int = 64):
        """
        Args:
            dim: Embedding dimension (Appendix D.2: 64).
        """
        self.dim = dim
        logger.warning("Using HashingEmbedder (64-dim). Semantic quality is limited. "
                        "Use SentenceTransformerEmbedder for production.")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts via SHA-256 token hashing, L2 normalized.

        Returns np.ndarray of shape [N, dim] with unit-norm vectors.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        results = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = text.lower().split()
            for token in tokens:
                digest = hashlib.sha256(token.encode()).digest()
                for i, byte in enumerate(digest[: self.dim]):
                    vec[i % self.dim] += byte / 255.0
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            results.append(vec)

        return np.stack(results, axis=0)

    def __repr__(self) -> str:
        return f"HashingEmbedder(dim={self.dim})"


def get_embedder(model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu") -> EmbedderBase:
    """
    Factory function: returns SentenceTransformerEmbedder if available, else HashingEmbedder.

    Paper reference: Appendix D.2 — "All experiments use SentenceTransformerEmbedder"
    """
    try:
        return SentenceTransformerEmbedder(model_name=model_name, device=device)
    except ImportError:
        logger.warning("sentence-transformers not available. Falling back to HashingEmbedder.")
        return HashingEmbedder()
