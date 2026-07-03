# Vector Databases and Embeddings

## What is an embedding?

An embedding is a dense numeric vector (typically 384 to 3072 dimensions) that
represents the meaning of a piece of text. Texts with similar meaning map to nearby
points in the vector space. Embeddings are produced by transformer encoder models
trained with contrastive objectives so that semantically related pairs end up close
together under cosine similarity.

Popular embedding models include sentence-transformers models such as all-MiniLM-L6-v2
(384 dimensions, fast, runs locally on CPU), OpenAI text-embedding-3 models, and
Cohere embed models. Model choice trades off accuracy, dimensionality, cost, and
latency.

## Approximate nearest neighbor search

Finding the exact nearest vectors among millions requires comparing against every
vector, which is too slow. Vector databases use approximate nearest neighbor (ANN)
indexes that trade a tiny amount of recall for orders-of-magnitude speedups:

- **HNSW (Hierarchical Navigable Small World)**: a multi-layer graph where search hops
  between neighbors, descending layers. Excellent recall-latency trade-off; the default
  choice in most vector databases.
- **IVF (Inverted File)**: clusters vectors with k-means, searches only the closest
  clusters. Memory-efficient for very large collections.
- **Product Quantization (PQ)**: compresses vectors into compact codes, reducing memory
  8-64x at some recall cost. Often combined with IVF (IVF-PQ).

## FAISS

FAISS (Facebook AI Similarity Search) is an open-source library for efficient vector
search. IndexFlatIP performs exact inner-product search — perfect for corpora up to a
few hundred thousand vectors. For larger corpora, IndexHNSWFlat or IndexIVFPQ provide
ANN search. With normalized vectors, inner product equals cosine similarity. FAISS is a
library, not a server: it has no built-in persistence, filtering, or replication, which
is what managed vector databases add on top.

## Managed vector databases

- **Pinecone**: fully managed, serverless, strong metadata filtering.
- **Qdrant**: open-source, Rust-based, rich payload filtering, on-prem friendly.
- **Milvus**: open-source, distributed, designed for billion-scale collections.
- **Weaviate**: open-source with built-in hybrid (BM25 + vector) search.
- **pgvector**: Postgres extension; convenient when you already run Postgres.

## Metadata filtering

Production retrieval almost always combines vector similarity with structured filters:
tenant id, document type, date range, or access permissions. Databases implement
pre-filtering (filter first, then search the survivors) or post-filtering (search
first, drop non-matching results). Pre-filtering guarantees k results but can be slow
on low-selectivity filters; post-filtering is fast but can return fewer than k results.
