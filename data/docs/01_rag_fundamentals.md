# Retrieval-Augmented Generation (RAG) Fundamentals

## What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that combines an information
retrieval system with a large language model (LLM). Instead of relying only on the
knowledge frozen inside the model's weights, a RAG system first retrieves relevant
documents from an external knowledge base, then passes those documents to the LLM as
context so it can generate a grounded, up-to-date answer.

RAG was introduced by Lewis et al. (2020) at Facebook AI Research. It has become the
standard approach for building enterprise question-answering systems because it reduces
hallucination, allows the knowledge base to be updated without retraining the model, and
makes answers auditable through source citations.

## Why RAG instead of fine-tuning?

Fine-tuning bakes knowledge into model weights. This is expensive, slow to update, and
hard to audit. RAG keeps knowledge in an external store that can be updated in seconds.
Key advantages of RAG over fine-tuning for knowledge injection:

- **Freshness**: add or remove documents instantly, no retraining required.
- **Traceability**: every answer can cite the exact source passages.
- **Cost**: indexing documents is far cheaper than training runs.
- **Access control**: retrieval can filter documents per user permission.

Fine-tuning is still the right tool for changing model *behavior* (style, format,
domain-specific reasoning), while RAG is the right tool for injecting *knowledge*.

## The canonical RAG pipeline

A production RAG pipeline has two phases:

1. **Ingestion (offline)**: documents are loaded, split into chunks, converted into
   dense vector embeddings, and stored in a vector database along with metadata.
2. **Query (online)**: the user question is embedded, the most similar chunks are
   retrieved, optionally re-ranked, and the top passages are inserted into a prompt
   that instructs the LLM to answer using only the provided context.

## Common failure modes

- **Retrieval miss**: the answer exists in the corpus but the retriever does not find
  it, often because of vocabulary mismatch between question and document.
- **Context overflow**: too many chunks are stuffed into the prompt, diluting the
  relevant signal and increasing cost.
- **Hallucination despite context**: the LLM ignores retrieved passages. Mitigated by
  explicit grounding instructions and by asking the model to cite sources.
- **Stale index**: documents changed but the index was not rebuilt.
