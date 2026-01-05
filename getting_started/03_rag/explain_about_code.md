# Code Explanation: Section 03 - RAG (Retrieval-Augmented Generation)

This file explains the RAG code examples, written for someone with data engineering background but new to RAG systems.

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Instead of asking an LLM to answer from its training data alone, RAG:
1. **Retrieves** relevant information from your documents
2. **Augments** the prompt with that information
3. **Generates** an answer using both the LLM's knowledge AND the retrieved context

**Why use RAG?**
- LLMs have training cutoff dates (they don't know recent information)
- LLMs can't access your private/internal documents
- RAG provides **grounded answers** (answers based on actual sources)
- Reduces hallucinations (made-up information)

**Analogy:** Like giving a student a textbook to reference during an exam, instead of relying only on memory.

---

## File 1: `basic_rag.py`

### The RAG Pipeline (5 Steps)

#### Step 1: Chunking

```python
def chunk_text(text, chunk_size=512, chunk_overlap=50):
```

**What is chunking?**
- Break long documents into smaller pieces (chunks)
- Why? Vector search works better on smaller, focused pieces
- Overlap prevents losing context at chunk boundaries

**Example:**
```
Document: "Machine learning is... [5000 characters]"
↓
Chunks:
  Chunk 1: "Machine learning is... [512 chars]"
  Chunk 2: "[overlap 50 chars]...deep learning... [512 chars]"
  Chunk 3: "[overlap 50 chars]...transformers... [512 chars]"
```

**Key parameters:**
- `chunk_size`: How big each chunk is (characters or tokens)
- `chunk_overlap`: How much overlap between chunks (prevents losing context)

**For data engineers:** Similar to partitioning a large table - you split it into manageable pieces.

---

#### Step 2: Embeddings

```python
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = embedding_model.encode(chunks)
```

**What are embeddings?**
- Convert text into **vectors** (arrays of numbers)
- Similar texts → similar vectors (close in vector space)
- Enables **semantic search** (find meaning, not just keywords)

**Example:**
```
"machine learning" → [0.2, -0.5, 0.8, ...] (vector of 384 numbers)
"deep learning"   → [0.3, -0.4, 0.7, ...] (similar vector)
"cooking recipe"  → [-0.1, 0.9, -0.2, ...] (very different vector)
```

**Why vectors?**
- Can calculate **distance** between vectors (cosine similarity)
- Closer vectors = more similar meaning
- Enables fast similarity search

**For data engineers:** Like creating a hash index, but for semantic similarity instead of exact matches.

---

#### Step 3: Vector Store

```python
collection = chromadb.Client().create_collection("rag_docs")
collection.add(documents=chunks, embeddings=embeddings)
```

**What is a vector store?**
- Database optimized for storing and searching vectors
- **Chroma**: Simple, local-first vector database
- **FAISS**: Facebook's vector search library (faster, more complex)

**What gets stored?**
- Document text (the chunk)
- Embedding vector
- Metadata (source, date, type, etc.)
- ID (unique identifier)

**For data engineers:** Like a NoSQL database, but optimized for vector similarity queries instead of key-value lookups.

---

#### Step 4: Retrieval

```python
query_embedding = embedding_model.encode([query])[0]
results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
```

**How retrieval works:**
1. Convert query to embedding (same model as documents)
2. Search vector store for most similar document embeddings
3. Return top-k most similar chunks

**Similarity calculation:**
- Usually **cosine similarity** (angle between vectors)
- Or **Euclidean distance** (straight-line distance)
- Lower distance / higher similarity = more relevant

**Example:**
```
Query: "What is machine learning?"
↓
Query embedding: [0.2, -0.5, 0.8, ...]
↓
Find chunks with closest embeddings:
  Chunk 1: distance 0.15 (very relevant!)
  Chunk 2: distance 0.45 (somewhat relevant)
  Chunk 3: distance 0.82 (not very relevant)
↓
Return top 3 chunks
```

**For data engineers:** Like a JOIN operation, but matching on semantic similarity instead of exact keys.

---

#### Step 5: Generation

```python
context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""
answer = call_ollama(prompt=prompt)
```

**What happens here?**
- Combine retrieved chunks into a "context" string
- Add context to the prompt
- Ask LLM to answer using the context

**Why this works:**
- LLM sees relevant information in the prompt
- Can cite specific sources
- Less likely to hallucinate (it has actual context)

**Prompt structure:**
```
Role: "You are a helpful assistant"
Context: [retrieved document chunks]
Question: [user's question]
Instructions: "Answer using only the context provided"
```

---

## File 2: `rag_with_reranking.py`

### What is Reranking?

**Problem with basic retrieval:**
- Vector search finds semantically similar chunks
- But "similar" doesn't always mean "relevant to the question"
- Example: Query "Python programming" might retrieve "Python snake facts" (both about Python, but different meanings)

**Solution: Reranking**
- Use a **more powerful model** to reorder retrieved chunks
- CrossEncoder: Takes (query, chunk) pairs and scores relevance
- More accurate but slower than vector search

**Two-stage retrieval:**
1. **Retrieve more** (e.g., top 10) using fast vector search
2. **Rerank** using slower but more accurate model
3. **Take top-k** after reranking (e.g., top 3)

**Trade-off:**
- ✅ Better relevance
- ❌ Slower (reranking model is expensive)
- 💡 Solution: Only rerank top candidates (not all documents)

---

### Reranking Code

```python
reranker = CrossEncoder("BAAI/bge-reranker-base")
pairs = [[query, chunk['text']] for chunk in chunks]
scores = reranker.predict(pairs)
```

**How it works:**
1. Create pairs: (query, chunk_text) for each retrieved chunk
2. Reranker scores each pair (higher = more relevant)
3. Sort chunks by rerank score
4. Take top-k after reranking

**Why CrossEncoder?**
- **BiEncoder** (embedding model): Encodes query and chunk separately, then compares
  - Fast, but less accurate
  - Used in initial retrieval
  
- **CrossEncoder**: Encodes query and chunk together
  - Slower, but more accurate
  - Used for reranking

**For data engineers:** Like a two-stage query:
1. Fast index scan (vector search)
2. Detailed evaluation of candidates (reranking)

---

## File 3: `rag_with_metadata_filtering.py`

### What is Metadata Filtering?

**Problem:**
- Sometimes you want to filter by source, date, type, etc.
- Example: "Only search in documents from 2024" or "Only search in API docs"

**Solution: Metadata**
- Store additional information with each chunk
- Filter during retrieval
- Like SQL WHERE clauses, but for vector search

---

### Metadata Structure

```python
metadata = {
    "source": "ml_basics.txt",
    "date": "2024-01-15",
    "type": "tutorial"
}
```

**Common metadata fields:**
- `source`: Which document this chunk came from
- `date`: When the document was created/updated
- `type`: Document type (article, tutorial, API doc, etc.)
- `author`: Who wrote it
- `category`: Topic category

**For data engineers:** Like adding columns to a table for filtering.

---

### Filtering Code

```python
where_clause = {"source": "ml_basics.txt"}
results = collection.query(
    query_embeddings=[query_embedding],
    where=where_clause
)
```

**How filtering works:**
1. Build `where` clause (like SQL WHERE)
2. Vector store filters chunks before/after similarity search
3. Returns only chunks matching the filter

**Filter types:**
- **Exact match**: `{"source": "doc.txt"}`
- **In list**: `{"type": {"$in": ["tutorial", "article"]}}`
- **Date range**: Filter after retrieval (Chroma's date filtering is limited)

**For data engineers:** Like adding WHERE conditions to a SELECT query.

---

## File 4: `evaluate_rag.py`

### Why Evaluate RAG?

**Problem:** How do you know if your RAG system is good?

**Solution: Evaluation metrics**

---

### Metric 1: Grounding Rate

```python
def check_grounding(answer, retrieved_chunks, expected_source):
```

**What is grounding?**
- Answer is **based on** the retrieved chunks
- Not hallucinated or from LLM's training data

**How to check:**
- Simple: Check if key words from answer appear in chunks
- Advanced: Use LLM-as-judge to check if answer is supported by context

**Why it matters:**
- RAG's main benefit is providing grounded answers
- If answers aren't grounded, RAG isn't working

---

### Metric 2: Answer Correctness

```python
def exact_match(predicted, expected):
def calculate_f1_score(predicted, expected):
```

**Exact match:**
- Predicted answer exactly matches expected (case-insensitive)
- Simple but strict (misses semantically correct but differently worded answers)

**F1 score:**
- Measures overlap between predicted and expected
- More forgiving than exact match
- Good for evaluating semantic similarity

**F1 calculation:**
```
Precision = (matching words) / (words in predicted answer)
Recall = (matching words) / (words in expected answer)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**For data engineers:** Like comparing two strings, but accounting for semantic similarity.

---

### Metric 3: Latency

```python
start_time = time.time()
answer = query_rag(...)
latency = time.time() - start_time
```

**What to measure:**
- **Average latency**: Mean response time
- **P95 latency**: 95th percentile (95% of requests are faster than this)
- **P99 latency**: 99th percentile (for production systems)

**Why it matters:**
- Users expect fast responses (< 2-3 seconds)
- RAG adds overhead (embedding, retrieval, reranking)
- Need to optimize for production

**For data engineers:** Like query performance metrics in databases.

---

### Evaluation Dataset

```python
eval_dataset = [
    {
        "question": "What is machine learning?",
        "expected_answer": "Machine learning is...",
        "expected_source": "doc_0",
        "context": "Machine learning is..."
    }
]
```

**What you need:**
- Questions (what users will ask)
- Expected answers (ground truth)
- Expected sources (which document should be retrieved)
- Context (documents to build RAG from)

**How many examples?**
- Start with 20-50 for initial testing
- Add more as you find edge cases
- Production systems: 100+ examples

**For data engineers:** Like a test dataset with expected outputs.

---

## Common Questions

### Q: How do I choose chunk size?

**Guidelines:**
- **Small chunks (256-512 tokens)**: Better for precise retrieval, but may lose context
- **Medium chunks (512-1024 tokens)**: Good balance (recommended starting point)
- **Large chunks (1024+ tokens)**: More context, but less precise retrieval

**Start with 512 tokens, adjust based on your documents.**

---

### Q: Which embedding model should I use?

**Options:**
- **OpenAI `text-embedding-3-small`**: Best quality, but requires API key (costs money)
- **BAAI/bge-small-en-v1.5**: Good quality, free, runs locally (recommended for learning)
- **BAAI/bge-large-en-v1.5**: Better quality, but slower and larger

**For learning:** Start with `bge-small-en-v1.5` (free, good quality).

---

### Q: Chroma vs FAISS?

**Chroma:**
- ✅ Easy to use, good for learning
- ✅ Built-in metadata filtering
- ✅ Persists to disk
- ❌ Slower for very large datasets

**FAISS:**
- ✅ Very fast (Facebook's library)
- ✅ Handles billions of vectors
- ❌ More complex API
- ❌ No built-in metadata filtering

**For learning:** Start with Chroma. Switch to FAISS if you need speed/scale.

---

### Q: When should I use reranking?

**Use reranking when:**
- Initial retrieval quality isn't good enough
- You have compute budget for slower queries
- Precision is more important than speed

**Skip reranking when:**
- Initial retrieval is already good
- You need very fast responses
- You're just learning (start simple!)

---

### Q: How do I improve RAG quality?

**Common improvements:**
1. **Better chunking**: Use semantic chunking (split on sentences/paragraphs, not just size)
2. **Better embeddings**: Use larger/better embedding models
3. **Reranking**: Add reranker for better relevance
4. **Hybrid search**: Combine vector search with keyword search (BM25)
5. **Query expansion**: Rewrite/expand user queries before retrieval
6. **Better prompts**: Improve the prompt that combines context + question

**Start simple, add complexity as needed.**

---

## Next Steps

After understanding these basics:

1. **Try different chunk sizes** - see how it affects retrieval
2. **Experiment with different embedding models** - compare quality
3. **Add reranking** - measure improvement vs overhead
4. **Build your own eval dataset** - test on your actual use case
5. **Optimize for production** - caching, batching, etc.

Then move to section 04 (Fine-tuning) to customize models for your specific domain!

