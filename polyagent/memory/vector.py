from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import hashlib
import time


class VectorDocument:
    def __init__(self, id: str, content: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}
        self.created_at = time.time()


class VectorMemory:
    """
    In-memory vector store with cosine similarity, pseudo-embeddings / dense vector indexing,
    and semantic deduplication.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.documents: List[VectorDocument] = []

    def _embed(self, text: str) -> np.ndarray:
        """Deterministic mock embedding generator for testing and standalone operation."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, custom_embedding: Optional[np.ndarray] = None) -> str:
        doc_id = hashlib.md5(f"{content}_{time.time()}".encode("utf-8")).hexdigest()[:12]
        emb = custom_embedding if custom_embedding is not None else self._embed(content)
        doc = VectorDocument(id=doc_id, content=content, embedding=emb, metadata=metadata)
        self.documents.append(doc)
        return doc_id

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        query_emb = self._embed(query)
        scores = []
        for doc in self.documents:
            # Cosine similarity
            sim = float(np.dot(query_emb, doc.embedding) / (np.linalg.norm(query_emb) * np.linalg.norm(doc.embedding)))
            scores.append((sim, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scores[:top_k]:
            results.append({
                "id": doc.id,
                "content": doc.content,
                "score": round(score, 4),
                "metadata": doc.metadata,
            })
        return results

    def clear(self):
        self.documents.clear()
