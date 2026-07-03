"""Hybrid retrieval: dense (FAISS) + sparse (BM25) -> RRF fusion -> cross-encoder re-ranking.

This is the heart of the project and mirrors how production RAG retrieval works:

  query ─┬─> dense search (FAISS, cosine)  ─┐
         │                                  ├─> Reciprocal Rank Fusion ─> cross-encoder
         └─> sparse search (BM25, lexical) ─┘        (rank-based merge)      re-ranker
                                                                              │
                                                                     final top-k chunks
"""

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from . import vectorstore
from .config import (
    CANDIDATES_PER_RETRIEVER,
    FINAL_TOP_K,
    INDEX_DIR,
    RERANKER_MODEL,
    RRF_K,
)
from .embeddings import embed_query

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    source: str
    section: str
    score: float                       # final score (rerank score, or fused score)
    retrievers: list[str] = field(default_factory=list)  # which retrievers found it

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.section,
            "score": round(self.score, 4),
            "retrievers": self.retrievers,
            "text": self.text,
        }


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


class HybridRetriever:
    """Loads the persisted index once and serves hybrid queries."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index, self.meta = vectorstore.load_index(index_dir)
        corpus_tokens = [_tokenize(m["text"]) for m in self.meta]
        self.bm25 = BM25Okapi(corpus_tokens)

    # --- individual retrievers -------------------------------------------

    def dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return vectorstore.search(self.index, embed_query(query), top_k)

    def sparse_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:top_k] if scores[i] > 0]

    # --- fusion + re-ranking ----------------------------------------------

    @staticmethod
    def _rrf_fuse(result_lists: dict[str, list[tuple[int, float]]]) -> dict[int, dict]:
        """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (RRF_K + rank)."""
        fused: dict[int, dict] = {}
        for name, results in result_lists.items():
            for rank, (chunk_id, _score) in enumerate(results):
                entry = fused.setdefault(chunk_id, {"score": 0.0, "retrievers": []})
                entry["score"] += 1.0 / (RRF_K + rank + 1)
                entry["retrievers"].append(name)
        return fused

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        use_hybrid: bool = True,
        use_reranker: bool = True,
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        """Full retrieval pipeline. Returns (chunks, per-stage timings in seconds)."""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        dense = self.dense_search(query, CANDIDATES_PER_RETRIEVER)
        timings["dense_search"] = time.perf_counter() - t0

        if use_hybrid:
            t0 = time.perf_counter()
            sparse = self.sparse_search(query, CANDIDATES_PER_RETRIEVER)
            timings["bm25_search"] = time.perf_counter() - t0
            fused = self._rrf_fuse({"dense": dense, "bm25": sparse})
        else:
            fused = self._rrf_fuse({"dense": dense})

        candidates = sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)

        results = [
            RetrievedChunk(
                chunk_id=cid,
                text=self.meta[cid]["text"],
                source=self.meta[cid]["source"],
                section=self.meta[cid]["section"],
                score=info["score"],
                retrievers=info["retrievers"],
            )
            for cid, info in candidates
        ]

        if use_reranker and results:
            t0 = time.perf_counter()
            reranker = _get_reranker()
            pairs = [(query, r.text) for r in results]
            rerank_scores = reranker.predict(pairs)
            for r, s in zip(results, rerank_scores):
                r.score = float(s)
            results.sort(key=lambda r: r.score, reverse=True)
            timings["rerank"] = time.perf_counter() - t0

        return results[:top_k], timings
