# Corrective RAG (CRAG)

## Definition
Corrective Retrieval-Augmented Generation (CRAG) is an extension of RAG that adds a self-correction step between retrieval and generation. It was introduced by Yan et al. in the 2024 paper "Corrective Retrieval Augmented Generation". The key idea is to evaluate the quality of retrieved documents before using them, and to take corrective action when the retrieval is weak.

## The retrieval evaluator
A lightweight retrieval evaluator scores every retrieved chunk for relevance to the question. Based on the scores, CRAG assigns one of three confidence levels:
- Correct: at least one chunk is clearly relevant. The relevant chunks are refined and used for generation.
- Ambiguous: the evaluator is not confident. CRAG combines the partially relevant chunks with additional retrieval.
- Incorrect: all chunks are irrelevant. They are discarded and the system searches again.

## Corrective actions
When the context is weak, the system rewrites the user's query into a better search query, for example by expanding abbreviations, adding synonyms or removing conversational filler. It then retrieves again with the rewritten query. The original paper also uses web search as an extra knowledge source; this project instead re-queries the uploaded document collection, with a limited retry budget.

## Knowledge refinement
CRAG uses a decompose-then-recompose algorithm: retrieved documents are split into small knowledge strips, irrelevant strips are filtered out, and the remaining strips are recombined. This keeps only the information that is useful for the answer.

## Benefits
CRAG reduces hallucination because the generator only receives verified context. When no relevant context can be found after the retries, the system declines to answer instead of guessing. CRAG is plug-and-play: it can be added on top of any existing RAG pipeline.
