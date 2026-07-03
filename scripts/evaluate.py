"""Retrieval evaluation: hit-rate@k ablation across retriever configurations.

For each test question we know which document contains the answer. We measure
how often that document appears in the top-k retrieved chunks, comparing:

  1. dense-only            (FAISS cosine similarity)
  2. hybrid (dense + BM25) (RRF fusion, no re-ranking)
  3. hybrid + re-ranking   (full pipeline)

Run from the project root:
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.retrieval import HybridRetriever

# (question, expected source document) — the doc that contains the answer.
TEST_SET: list[tuple[str, str]] = [
    ("What is Reciprocal Rank Fusion and what value of k is commonly used?", "03_hybrid_search.md"),
    ("Why does BM25 outperform embeddings on product codes and error messages?", "03_hybrid_search.md"),
    ("What is the difference between a bi-encoder and a cross-encoder?", "04_reranking.md"),
    ("Which model family is commonly used for open-source re-ranking?", "04_reranking.md"),
    ("How does HNSW indexing work?", "05_vector_databases.md"),
    ("When should I use pgvector instead of a dedicated vector database?", "05_vector_databases.md"),
    ("What chunk size should I use for question answering?", "02_chunking_strategies.md"),
    ("What is parent-child or small-to-big chunking?", "02_chunking_strategies.md"),
    ("How do I reduce hallucination in grounded generation prompts?", "06_llm_generation.md"),
    ("What temperature should be used for factual RAG answers?", "06_llm_generation.md"),
    ("What is hit rate at k and why does it matter?", "07_rag_evaluation.md"),
    ("How is faithfulness of a generated answer measured?", "07_rag_evaluation.md"),
    ("What should a RAG service do when the LLM provider is down?", "08_production_rag.md"),
    ("How does blue-green index deployment work?", "08_production_rag.md"),
    ("Why choose RAG over fine-tuning for knowledge injection?", "01_rag_fundamentals.md"),
    ("What are the two phases of the canonical RAG pipeline?", "01_rag_fundamentals.md"),
]

TOP_K = 3


def evaluate(
    retriever: HybridRetriever, use_hybrid: bool, use_reranker: bool
) -> tuple[float, float]:
    """Returns (hit_rate@k, MRR@k).

    MRR rewards ranking the correct source *first*, not just somewhere in top-k —
    this is where re-ranking shows its value.
    """
    hits = 0
    reciprocal_ranks = 0.0
    for question, expected_source in TEST_SET:
        chunks, _ = retriever.retrieve(
            question, top_k=TOP_K, use_hybrid=use_hybrid, use_reranker=use_reranker
        )
        for rank, c in enumerate(chunks, start=1):
            if c.source == expected_source:
                hits += 1
                reciprocal_ranks += 1.0 / rank
                break
    n = len(TEST_SET)
    return hits / n, reciprocal_ranks / n


def main() -> None:
    retriever = HybridRetriever()
    configs = [
        ("dense-only", False, False),
        ("hybrid (dense+BM25)", True, False),
        ("hybrid + re-ranking", True, True),
    ]
    print(f"\nRetrieval ablation over {len(TEST_SET)} questions\n")
    print(f"{'configuration':<24} {'hit-rate@' + str(TOP_K):>10} {'MRR@' + str(TOP_K):>8}")
    print("-" * 46)
    for name, use_hybrid, use_reranker in configs:
        rate, mrr = evaluate(retriever, use_hybrid, use_reranker)
        print(f"{name:<24} {rate:>9.1%} {mrr:>8.3f}")
    print()


if __name__ == "__main__":
    main()
