"""Build the search indexes: docs -> semantic chunks -> embeddings -> FAISS.

Run from the project root:
    python scripts/build_index.py
"""

import sys
import time
from pathlib import Path

# Make `src/` importable when running as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import vectorstore
from rag.chunking import chunk_directory
from rag.config import DOCS_DIR, INDEX_DIR
from rag.embeddings import embed_texts


def main() -> None:
    t0 = time.perf_counter()

    print(f"[1/3] Chunking documents in {DOCS_DIR} ...")
    chunks = chunk_directory(DOCS_DIR)
    n_docs = len({c.source for c in chunks})
    print(f"      {n_docs} documents -> {len(chunks)} chunks")

    print("[2/3] Embedding chunks (sentence-transformers, local CPU) ...")
    embeddings = embed_texts([c.text for c in chunks])
    print(f"      embeddings shape: {embeddings.shape}")

    print(f"[3/3] Building FAISS index -> {INDEX_DIR} ...")
    vectorstore.build_index(chunks, embeddings, INDEX_DIR)

    print(f"Done in {time.perf_counter() - t0:.1f}s. "
          f"Index files written to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
