"""Streamlit chat UI for the RAG demo.

Run (with the FastAPI server already running on port 8000):
    streamlit run ui/app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Advanced RAG Demo", page_icon="🔎", layout="wide")

st.title("🔎 Advanced RAG — Hybrid Search + Re-ranking")
st.caption(
    "Dense (FAISS) + BM25 retrieval → Reciprocal Rank Fusion → "
    "cross-encoder re-ranking → DeepSeek generation with source citations."
)

# --- Sidebar: pipeline controls -------------------------------------------
with st.sidebar:
    st.header("Pipeline settings")
    top_k = st.slider("Chunks sent to LLM (top-k)", 1, 10, 5)
    use_hybrid = st.toggle("Hybrid search (dense + BM25)", value=True)
    use_reranker = st.toggle("Cross-encoder re-ranking", value=True)

    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API: online")
        st.write(f"LLM configured: {'✅' if health['llm_configured'] else '❌ (extractive fallback)'}")
    except requests.RequestException:
        st.error("API offline — start it with:\n`uvicorn api.main:app --port 8000`")

    st.divider()
    st.markdown(
        "**Try asking:**\n"
        "- What is hybrid search and why does it beat dense-only?\n"
        "- How does a cross-encoder re-ranker work?\n"
        "- What chunk size should I use for Q&A?\n"
        "- What should happen when the LLM provider is down?"
    )

# --- Chat history ----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("payload"):
            _payload = msg["payload"]
            with st.expander(f"📚 Sources ({len(_payload['sources'])}) & latency"):
                for i, s in enumerate(_payload["sources"], start=1):
                    st.markdown(
                        f"**[{i}] {s['source']}** — *{s['section']}* "
                        f"(score {s['score']}, found by: {', '.join(s['retrievers'])})"
                    )
                    st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))
                st.json(_payload["timings_ms"])

# --- Input -----------------------------------------------------------------
if question := st.chat_input("Ask a question about RAG / AI engineering..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving → fusing → re-ranking → generating..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "use_hybrid": use_hybrid,
                        "use_reranker": use_reranker,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        mode_badge = "🤖 DeepSeek" if payload["mode"] == "llm" else "📄 Extractive fallback"
        st.markdown(f"{payload['answer']}\n\n`{mode_badge}`")

        with st.expander(f"📚 Sources ({len(payload['sources'])}) & latency"):
            for i, s in enumerate(payload["sources"], start=1):
                st.markdown(
                    f"**[{i}] {s['source']}** — *{s['section']}* "
                    f"(score {s['score']}, found by: {', '.join(s['retrievers'])})"
                )
                st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))
            st.json(payload["timings_ms"])

    st.session_state.messages.append(
        {"role": "assistant", "content": payload["answer"], "payload": payload}
    )
