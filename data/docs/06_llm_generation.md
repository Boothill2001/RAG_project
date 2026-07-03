# LLM Generation and Prompting in RAG

## Grounded generation

The final step of a RAG pipeline inserts the retrieved passages into a prompt and asks
the LLM to answer using only that context. A robust grounding prompt includes:

- A system instruction defining the assistant's role and rules.
- The retrieved passages, each labeled with a source identifier.
- An explicit instruction to answer *only* from the provided context and to say
  "I don't know" when the context is insufficient.
- An instruction to cite the source identifiers used in the answer.

Explicitly allowing the model to refuse ("if the context does not contain the answer,
say so") dramatically reduces hallucination compared with prompts that pressure the
model to always produce an answer.

## Prompt structure example

A typical RAG prompt template looks like:

    System: You are a helpful assistant. Answer the user's question using ONLY the
    provided context. Cite sources as [1], [2]. If the answer is not in the context,
    say you don't know.

    Context:
    [1] (chunking_strategies.md) Semantic chunking splits documents along...
    [2] (hybrid_search.md) Reciprocal Rank Fusion combines ranked lists...

    Question: How should I split my documents?

## Temperature and determinism

For factual question answering, use a low temperature (0 to 0.3) so the model stays
close to the retrieved evidence. Higher temperatures increase creativity but also the
risk of drifting from the context. Structured outputs (JSON mode or schema-constrained
decoding) are useful when downstream code parses the answer.

## Streaming responses

Production chat interfaces stream tokens as they are generated (server-sent events or
chunked HTTP) so users see the answer forming immediately. Time-to-first-token is the
key perceived-latency metric; total generation time matters less once streaming starts.

## Cost and latency engineering

- Keep prompts tight: send the top 3-8 re-ranked passages, not everything retrieved.
- Cache identical or near-duplicate questions and their answers.
- Use provider prompt caching for long shared prefixes (system prompt plus corpus
  boilerplate) to cut input token costs.
- Route easy queries to a small cheap model and hard queries to a large model
  (model cascading), cutting average cost substantially.

## Evaluating generated answers

Common metrics: faithfulness (is every claim supported by the retrieved context?),
answer relevance (does it address the question?), and citation precision (do the cited
sources actually contain the claims?). Frameworks like RAGAS automate these with
LLM-as-judge evaluation.
