"""FastAPI service exposing the RAG pipeline.

Run from the project root:
    uvicorn api.main:app --reload --port 8000

Interactive API docs (Swagger UI): http://localhost:8000/docs
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEEPSEEK_API_KEY  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indexes once at startup, not per request."""
    global pipeline
    pipeline = RAGPipeline()
    # Warm-up: force lazy model loads (embedder + re-ranker) now, so the first
    # real request doesn't pay the multi-second cold-start cost.
    pipeline.retriever.retrieve("warmup query", top_k=1)
    yield


app = FastAPI(
    title="Advanced RAG API",
    description=(
        "Retrieval-Augmented Generation with hybrid search (dense + BM25), "
        "Reciprocal Rank Fusion, and cross-encoder re-ranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          examples=["What is hybrid search in RAG?"])
    top_k: int = Field(5, ge=1, le=10, description="Number of chunks given to the LLM")
    use_hybrid: bool = Field(True, description="Combine dense + BM25 (vs dense-only)")
    use_reranker: bool = Field(True, description="Apply cross-encoder re-ranking")


class Source(BaseModel):
    chunk_id: int
    source: str
    section: str
    score: float
    retrievers: list[str]
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str  # "llm" (DeepSeek) or "extractive" (fallback)
    sources: list[Source]
    timings_ms: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_loaded": pipeline is not None,
        "llm_configured": bool(DEEPSEEK_API_KEY),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    # The pipeline is CPU/network bound synchronous code; run it in the
    # threadpool so the event loop keeps serving other requests.
    return await run_in_threadpool(
        pipeline.ask,
        request.question,
        top_k=request.top_k,
        use_hybrid=request.use_hybrid,
        use_reranker=request.use_reranker,
    )
