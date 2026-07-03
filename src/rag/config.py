"""Central configuration for the RAG pipeline.

All tunable knobs live here so the rest of the code stays clean.
Secrets (the DeepSeek API key) are read from a `.env` file which is
NEVER committed to git — see `.env.example` for the expected format.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels up from this file (src/rag/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from the project root (silently does nothing if the file is absent)
load_dotenv(PROJECT_ROOT / ".env")

# --- Paths ---------------------------------------------------------------
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
INDEX_DIR = PROJECT_ROOT / "index"

# --- Chunking ------------------------------------------------------------
MAX_CHUNK_CHARS = 1200   # hard cap on chunk size (roughly ~300 tokens)
CHUNK_OVERLAP_CHARS = 150

# --- Models --------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, CPU-friendly
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Retrieval -----------------------------------------------------------
CANDIDATES_PER_RETRIEVER = 20  # top-k pulled from each of dense / BM25 before fusion
RRF_K = 60                     # constant in Reciprocal Rank Fusion
FINAL_TOP_K = 5                # chunks handed to the LLM after re-ranking

# --- LLM (DeepSeek is OpenAI-API compatible) -----------------------------
# Accept a couple of common variable spellings so setup mistakes don't bite.
def _clean_key(raw: str) -> str:
    """Extract the bare `sk-...` token, tolerating quotes/comments in .env values."""
    match = re.search(r"sk-[A-Za-z0-9]+", raw)
    return match.group(0) if match else raw.strip()


DEEPSEEK_API_KEY = _clean_key(
    os.getenv("DEEPSEEK_KEY")
    or os.getenv("deepseek_key")
    or os.getenv("DEEPSEEK_API_KEY")
    or ""
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.1          # low temperature = stay close to the retrieved evidence
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT_SECONDS = 60
