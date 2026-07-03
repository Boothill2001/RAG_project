"""FAISS vector store: build, persist, load, and search.

We use IndexFlatIP (exact inner-product search). With L2-normalized vectors,
inner product == cosine similarity. Exact search is the right choice at this
corpus size; for millions of vectors you would switch to IndexHNSWFlat.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk

_INDEX_FILE = "chunks.faiss"
_META_FILE = "chunks.json"


def build_index(chunks: list[Chunk], embeddings: np.ndarray, index_dir: Path) -> None:
    """Build a FAISS index over chunk embeddings and persist it with metadata."""
    index_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / _INDEX_FILE))

    meta = [c.to_dict() for c in chunks]
    (index_dir / _META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_index(index_dir: Path) -> tuple["faiss.Index", list[dict]]:
    """Load the FAISS index and chunk metadata from disk."""
    index_path = index_dir / _INDEX_FILE
    meta_path = index_dir / _META_FILE
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}. Run `python scripts/build_index.py` first."
        )
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta


def search(index: "faiss.Index", query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Return [(chunk_id, score), ...] for the top_k most similar chunks."""
    scores, ids = index.search(query_vector, top_k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
