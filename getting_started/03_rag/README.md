## 3. Retrieval-Augmented Generation (RAG) (3–4 weeks)

### Goal
Build and evaluate a minimal RAG system.

### Learn
- Chunking strategies (size, overlap)
- Embeddings and vector stores (Chroma or FAISS)
- Reranking and metadata filtering
- Basic RAG evaluation metrics

### Do
- Build RAG v0:
  - Chunk a small document set
  - Embed chunks (`text-embedding-3-small` or `bge-small`)
  - Store in Chroma or FAISS
  - Retrieve top-k chunks for a query and feed them to an LLM
- Improve:
  - Add a reranker (`bge-reranker-base`)
  - Add metadata filters (source, recency, type)
- Evaluate:
  - Create 20–50 labeled Q/A pairs with references
  - Measure:
    - Grounding rate (% answers with correct cited source)
    - Answer correctness (exact match or F1)
    - Latency

### Done when
- Your RAG system clearly outperforms a non-RAG baseline on your eval set.
- You have a script/notebook that runs RAG + evaluation end-to-end.

