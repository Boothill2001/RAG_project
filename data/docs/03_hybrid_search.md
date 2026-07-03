# Hybrid Search: Dense + Sparse Retrieval

## Two families of retrieval

Modern search systems combine two complementary retrieval paradigms:

- **Sparse (lexical) retrieval** represents text as a bag of weighted terms. BM25 is
  the dominant algorithm: it scores documents by term frequency, inverse document
  frequency, and document length normalization. Sparse retrieval excels at exact
  keyword matches, rare terms, product codes, error messages, and names.
- **Dense (semantic) retrieval** encodes text into fixed-size embedding vectors using
  a neural network, then finds nearest neighbors by cosine similarity or dot product.
  Dense retrieval excels at paraphrases and synonyms — a query about "reducing model
  latency" can match a document about "speeding up inference" even with zero shared
  keywords.

## Why hybrid beats either alone

Dense retrieval fails on out-of-vocabulary tokens like SKUs, acronyms, or error codes
that the embedding model never learned. Sparse retrieval fails when the user's wording
differs from the document's wording. Hybrid search runs both retrievers and fuses their
results, capturing exact matches and semantic matches simultaneously. On enterprise
benchmarks, hybrid retrieval typically improves recall by 10-25% over dense-only.

## BM25 in detail

BM25 scores a document D for query Q as the sum over query terms of
IDF(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D| / avgdl)). The parameter k1
(typically 1.2-2.0) controls term-frequency saturation and b (typically 0.75) controls
length normalization. BM25 requires no training and is extremely fast with an inverted
index.

## Fusion methods

After running both retrievers you must merge two ranked lists:

- **Reciprocal Rank Fusion (RRF)**: score(d) = sum over each list of 1 / (k + rank(d)),
  with k commonly set to 60. RRF ignores raw scores entirely and only uses ranks, which
  makes it robust to score scale differences between BM25 and cosine similarity. It is
  the most widely used fusion method in production.
- **Weighted score fusion**: normalize both score distributions (min-max or z-score)
  and combine as alpha * dense + (1 - alpha) * sparse. Requires tuning alpha per corpus.

## Practical implementation

A minimal production hybrid retriever needs: a FAISS (or similar) vector index for the
dense side, a BM25 index over tokenized chunks for the sparse side, and an RRF fusion
step. Retrieve the top 20-50 candidates from each side, fuse, then pass the fused top-k
to a re-ranker for final ordering.
