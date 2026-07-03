# Production RAG: Architecture and Operations

## Serving architecture

A production RAG service typically exposes an HTTP API (FastAPI is the de facto
standard in Python) in front of the pipeline. Key API design points:

- **Async endpoints**: retrieval and LLM calls are I/O bound; async handlers let one
  worker serve many concurrent requests.
- **Pydantic schemas**: validate request and response payloads, and auto-generate
  OpenAPI documentation.
- **Health and readiness probes**: /health endpoints that verify the index is loaded
  and the LLM provider is reachable, used by load balancers and Kubernetes.
- **Timeouts and retries**: LLM providers fail; wrap calls with timeouts, exponential
  backoff, and a circuit breaker.

## Graceful degradation

A resilient RAG service defines fallback behavior for every dependency failure:

- LLM provider down or out of quota → return an extractive answer built from the
  top retrieved passages, clearly labeled as such.
- Re-ranker slow → skip re-ranking and serve fused retrieval order.
- Vector index unavailable → fall back to BM25-only retrieval.

Users receive a degraded but useful answer instead of an error page.

## Caching layers

- **Embedding cache**: identical texts should never be embedded twice.
- **Retrieval cache**: cache fused candidate lists for popular queries.
- **Answer cache**: exact or semantic (embedding-similarity) match on recent questions.
- **Provider prompt cache**: reuse the long static prefix across calls.

## Index lifecycle

Documents change. Production systems rebuild or incrementally update indexes on a
schedule or via change-data-capture. Blue-green index deployment — build the new index
alongside the old, run smoke queries, then atomically swap — avoids serving partial
indexes. Version every index with the embedding model name; mixing embeddings from
different models in one index silently corrupts similarity scores.

## Security and multi-tenancy

- Filter retrieval by tenant and user permissions *before* passages reach the prompt;
  the LLM must never see documents the user cannot read.
- Sanitize retrieved content against prompt injection: a malicious document can carry
  instructions like "ignore previous instructions". Mitigations include instruction
  hierarchy in prompts, content scanning, and marking retrieved text as data.
- Never log secrets; redact PII in stored traces where regulations require it.

## Cost tracking

Track per-request token usage (input, output, cached) and attribute costs per feature
and per tenant. Dashboards of cost per answered question guide model routing decisions
and cache tuning.
