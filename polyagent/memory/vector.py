"""In-memory vector memory with deterministic feature hashing and semantic deduplication."""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections import Counter
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class VectorDocument:
    """Indexed document with embedding vector and optional metadata."""

    def __init__(
        self,
        id: str,
        content: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.content = content
        self.embedding = embedding
        self.metadata = dict(metadata) if metadata else {}
        self.created_at = time.time()


class VectorMemory:
    """In-memory vector store with deterministic feature hashing, filtering, and deduplication."""

    def __init__(self, dim: int = 128, dedup_threshold: float = 0.985) -> None:
        if dim <= 0:
            raise ValueError("Vector dimension must be positive")
        self.dim = dim
        self.dedup_threshold = dedup_threshold
        self.documents: list[VectorDocument] = []

    def _bucket_index(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little") % self.dim

    def _embed(self, text: str) -> np.ndarray:
        """Compute an L2-normalized embedding from word unigrams and char trigrams."""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())

        if tokens:
            for token, freq in Counter(tokens).items():
                weight = 1.0 + math.log(freq)
                vec[self._bucket_index(f"w:{token}")] += weight
                if len(token) >= 3:
                    padded = f"^{token}$"
                    trigram_w = 0.35 * weight
                    for i in range(len(padded) - 2):
                        vec[self._bucket_index(f"c:{padded[i : i + 3]}")] += trigram_w
        else:
            cleaned = text.lower().strip()
            for i in range(max(0, len(cleaned) - 2)):
                vec[self._bucket_index(f"raw:{cleaned[i : i + 3]}")] += 1.0

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        custom_embedding: np.ndarray | None = None,
        deduplicate: bool = True,
    ) -> str:
        """Store a document, returning existing id if a near-duplicate already exists."""
        emb = (
            np.asarray(custom_embedding, dtype=np.float32)
            if custom_embedding is not None
            else self._embed(content)
        )

        if deduplicate:
            for existing in self.documents:
                if existing.content == content or (
                    self._cosine_sim(emb, existing.embedding) >= self.dedup_threshold
                ):
                    return existing.id

        doc_id = uuid.uuid4().hex[:12]
        doc = VectorDocument(id=doc_id, content=content, embedding=emb, metadata=metadata)
        self.documents.append(doc)
        return doc_id

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search documents ranked by cosine similarity with optional metadata filters."""
        if not self.documents or top_k <= 0:
            return []

        query_emb = self._embed(query)
        scored: list[tuple[float, VectorDocument]] = []
        for doc in self.documents:
            if filters and not all(doc.metadata.get(k) == v for k, v in filters.items()):
                continue
            scored.append((self._cosine_sim(query_emb, doc.embedding), doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "id": doc.id,
                "content": doc.content,
                "score": round(score, 4),
                "metadata": dict(doc.metadata),
            }
            for score, doc in scored[:top_k]
        ]

    def delete(self, doc_id: str) -> bool:
        """Remove a document by id, returning True if deleted."""
        for idx, doc in enumerate(self.documents):
            if doc.id == doc_id:
                self.documents.pop(idx)
                return True
        return False

    def clear(self) -> None:
        """Remove all stored documents."""
        self.documents.clear()
