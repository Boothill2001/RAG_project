# Chunking Strategies for RAG

## Why chunking matters

Embedding models and LLM context windows both have size limits, so documents must be
split into smaller pieces called chunks before indexing. Chunking quality directly
determines retrieval quality: chunks that are too small lose context, while chunks that
are too large dilute the embedding signal and waste prompt tokens.

## Fixed-size chunking

The simplest strategy splits text every N characters or tokens, usually with an overlap
of 10-20% so that sentences cut at a boundary still appear intact in the neighboring
chunk. Fixed-size chunking is fast and predictable but frequently cuts through the
middle of sentences, tables, or code blocks, producing incoherent fragments.

## Semantic chunking

Semantic chunking splits documents along natural meaning boundaries instead of raw
character counts. Practical implementations use:

- **Structural boundaries**: markdown headings, paragraphs, list items, and code fences.
- **Embedding-based boundaries**: compute embeddings for consecutive sentences and cut
  where cosine similarity between adjacent sentences drops sharply, indicating a topic
  shift.

Semantic chunking keeps each chunk about a single coherent topic, which produces
sharper embeddings and better retrieval precision. The trade-off is variable chunk
sizes, so a maximum size cap with recursive splitting is usually applied on top.

## Recursive character splitting

A popular middle ground (used by LangChain's RecursiveCharacterTextSplitter) tries a
priority list of separators: first split on double newlines (paragraphs), then single
newlines, then sentences, then words, until every chunk fits under the size limit. This
respects document structure when possible and only degrades to hard cuts when forced.

## Chunk size guidelines

- Typical chunk sizes range from 200 to 800 tokens.
- Question-answering over factual documents works well with 200-400 token chunks.
- Summarization and reasoning tasks benefit from larger 500-1000 token chunks.
- Overlap of 50-100 tokens preserves context across boundaries.
- Always store metadata with each chunk: source file, section title, and position,
  so answers can cite their origin.

## Parent-child chunking

An advanced pattern indexes small chunks for precise retrieval but returns the larger
parent section to the LLM. Small chunks match queries precisely; the parent provides
the surrounding context the model needs to answer well. This is also called
"small-to-big" retrieval.
