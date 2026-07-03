"""RAGPipeline: the single entry point that ties retrieval and generation together."""

from .generation import generate_answer
from .retrieval import HybridRetriever


class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()

    def ask(
        self,
        question: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_reranker: bool = True,
    ) -> dict:
        """Answer a question over the knowledge base.

        Returns a dict with the answer, the source chunks used, the generation
        mode ("llm" or "extractive") and per-stage latency — everything a demo
        UI or an API response needs.
        """
        chunks, timings = self.retriever.retrieve(
            question, top_k=top_k, use_hybrid=use_hybrid, use_reranker=use_reranker
        )
        answer, mode, gen_seconds = generate_answer(question, chunks)
        timings["generation"] = gen_seconds

        return {
            "question": question,
            "answer": answer,
            "mode": mode,
            "sources": [c.to_dict() for c in chunks],
            "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items()},
        }
