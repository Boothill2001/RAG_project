"""Unit tests for RRF fusion and config key parsing — no model downloads required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import RRF_K, _clean_key
from rag.retrieval import HybridRetriever


def test_rrf_fuses_two_ranked_lists():
    fused = HybridRetriever._rrf_fuse(
        {
            "dense": [(10, 0.9), (20, 0.8), (30, 0.7)],
            "bm25": [(20, 5.0), (40, 4.0)],
        }
    )
    # Chunk 20 appears in both lists -> highest fused score
    top = max(fused.items(), key=lambda kv: kv[1]["score"])
    assert top[0] == 20
    assert sorted(top[1]["retrievers"]) == ["bm25", "dense"]


def test_rrf_score_formula():
    fused = HybridRetriever._rrf_fuse({"dense": [(1, 0.5)]})
    assert abs(fused[1]["score"] - 1.0 / (RRF_K + 1)) < 1e-9


def test_rrf_ignores_raw_score_scale():
    """RRF must depend only on rank, not on the magnitude of raw scores."""
    small = HybridRetriever._rrf_fuse({"dense": [(1, 0.001), (2, 0.0005)]})
    large = HybridRetriever._rrf_fuse({"dense": [(1, 999.0), (2, 500.0)]})
    assert small[1]["score"] == large[1]["score"]
    assert small[2]["score"] == large[2]["score"]


def test_clean_key_extracts_token_from_messy_env_value():
    assert _clean_key("'sk-abc123 - work_api deepseek'") == "sk-abc123"
    assert _clean_key("sk-abc123") == "sk-abc123"
    assert _clean_key("") == ""
