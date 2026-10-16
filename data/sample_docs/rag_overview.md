# Retrieval-Augmented Generation (RAG) Overview

## What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that combines a retriever with a large language model (LLM). Instead of relying only on what the model memorised during training, the system first searches a document collection for passages related to the user's question and then passes those passages to the language model as context. The model writes its answer from that context.

## Why use RAG?
RAG lets an assistant answer questions about private or recent documents without retraining the model. It also makes answers traceable, because every answer can be linked back to the source passages it was generated from. Updating knowledge only requires re-indexing documents, which is much cheaper than fine-tuning.

## Standard RAG pipeline
1. Ingestion: documents are uploaded, text is extracted, cleaned and split into chunks.
2. Embedding: each chunk is converted into a dense vector by an embedding model such as all-MiniLM-L6-v2, which produces 384-dimensional vectors.
3. Indexing: vectors are stored in a vector database such as FAISS, Chroma or Pinecone.
4. Retrieval: the user's question is embedded and the top-k most similar chunks are returned using cosine similarity.
5. Generation: the retrieved chunks and the question are sent to the LLM, which produces the final answer.

## Limitations of standard RAG
Standard RAG trusts whatever the retriever returns. If the retrieved chunks are irrelevant or only loosely related, the language model may still try to answer and produce a hallucination: a fluent answer that is not supported by the documents. Poorly phrased questions, vocabulary mismatch between the question and the documents, and very short queries are common causes of weak retrieval.
