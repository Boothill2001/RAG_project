# Re-ranking in Retrieval Pipelines

## The two-stage retrieval pattern

Production search systems use a fast first-stage retriever to gather candidates and a
slower, more accurate second-stage re-ranker to order them. The first stage (BM25,
dense retrieval, or hybrid) scans millions of documents in milliseconds. The re-ranker
then examines only the top 20-100 candidates with a much more powerful model.

## Bi-encoders vs cross-encoders

- A **bi-encoder** embeds the query and the document independently into two vectors
  and compares them with cosine similarity. Because document vectors are precomputed,
  bi-encoders are fast enough for first-stage retrieval over huge corpora — but the
  query and document never see each other during encoding, limiting accuracy.
- A **cross-encoder** concatenates the query and document into a single input and runs
  a full transformer forward pass, producing a relevance score. Every query-document
  token pair can attend to each other, which makes cross-encoders significantly more
  accurate — but far too slow for first-stage retrieval, since nothing can be
  precomputed.

Re-ranking gets the best of both: bi-encoder (or BM25) speed for candidate generation,
cross-encoder accuracy for the final ordering.

## Typical models

Popular open-source cross-encoder re-rankers include the ms-marco MiniLM family
(for example cross-encoder/ms-marco-MiniLM-L-6-v2), trained on the MS MARCO passage
ranking dataset. Commercial options include Cohere Rerank and Voyage AI rerankers.
Late-interaction models such as ColBERT sit between the two families: they precompute
per-token document embeddings and perform a cheap MaxSim interaction at query time.

## Impact on RAG quality

In RAG, the re-ranker decides which passages actually enter the LLM prompt. Because
prompt space is limited, promoting the truly relevant passage from rank 15 to rank 2
can be the difference between a correct answer and a hallucination. Measured on
enterprise Q&A benchmarks, adding a cross-encoder re-ranker on top of hybrid retrieval
commonly lifts answer accuracy by 5-15 percentage points.

## Practical settings

- Retrieve 20-50 candidates from the first stage, re-rank, keep the top 3-8 for the prompt.
- Truncate document passages to the re-ranker's maximum sequence length (usually 512 tokens).
- Cache re-ranker scores for repeated query-document pairs.
- Monitor latency: a 6-layer MiniLM cross-encoder scores roughly 100-1000 pairs per
  second on CPU, so re-ranking 20 candidates adds only tens of milliseconds.
