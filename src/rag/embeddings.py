"""Dense embeddings via sentence-transformers (runs locally, no API cost).

all-MiniLM-L6-v2 produces 384-dim vectors, is fast on CPU, and is the standard
baseline model for English semantic search. Vectors are L2-normalized so that
FAISS inner-product search equals cosine similarity.
"""

from functools import lru_cache

import numpy as np

from .config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model():
    # Imported lazily so that importing this module stays cheap
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts -> (n, 384) float32 array, L2-normalized."""
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,   # normalized => inner product == cosine
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query -> (1, 384) float32 array."""
    return embed_texts([query])
