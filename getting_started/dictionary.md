# LLM Dictionary

This dictionary grows as you learn. Terms are added when you ask about them.

## Token

A **small unit of text** that the model reads or writes.

- Depending on the tokenizer, a token can be:
  - A whole word (`hello`)
  - A sub‑word piece (`trans`, `former`)
  - Punctuation or special symbols

- When we feed data to an LLM:
  1. Text → tokenizer → **sequence of token ids** (integers)
  2. Token ids → **embeddings** (vectors) → transformer layers.

- Example:
  - Text: `"Hello world!"`
  - Tokens (simplified): `["Hello", "world", "!"]`
  - Token ids (example): `[101, 203, 999]`  
    (the ids themselves don't matter, only that the model uses them consistently).

---

## Prompt

The **input text** you send to an LLM to get a response.

- Can be a simple question: `"What is machine learning?"`
- Or structured instructions with examples, context, and format requirements.
- **Prompt engineering** = designing effective prompts to get better results.

**Example:**
```
Classify this text as positive or negative.

Text: "I love this product!"
Sentiment:
```

---

## Zero-shot

Asking the LLM to do a task **without providing examples** beforehand.

- The model uses its pre-trained knowledge to understand what you want.
- Good for simple, common tasks.

**Example:**
```
Classify this text as positive, negative, or neutral.

Text: "This is great!"
```

---

## Few-shot

Providing **examples** in your prompt before asking the model to do the same task.

- Shows the model the pattern you want.
- Good for custom categories, specific formats, or complex patterns.

**Example:**
```
Classify customer feedback.

Examples:
Feedback: "The app crashes" → Category: Bug
Feedback: "Can you add dark mode?" → Category: Feature Request

Feedback: "How do I export data?" → Category:
```

---

## Chain-of-Thought (CoT)

Asking the model to **show its reasoning step by step** instead of jumping directly to the answer.

- Better for math problems, logic puzzles, complex reasoning.
- Helps you understand HOW the model got the answer.

**Example:**
```
Solve step by step: If a train travels 60 mph for 2 hours, how far does it go?

Step 1: Distance = Speed × Time
Step 2: Distance = 60 mph × 2 hours
Step 3: Distance = 120 miles
```

---

## Temperature

A parameter that controls **randomness** in the model's output.

- Range: `0.0` to `2.0`
- `0.0` = deterministic (same input → same output, very consistent)
- `1.0+` = very random/creative (same input → different outputs)

**When to use:**
- Low (0.0-0.3): Classification, extraction, code generation (need consistency)
- Medium (0.5-0.7): General chat, writing (balanced)
- High (0.8-2.0): Creative writing, brainstorming (want variety)

**Analogy:** Like adjusting how "creative" vs "precise" you want the model to be.

---

## Top_p (Nucleus Sampling)

Alternative to temperature for controlling randomness.

- Range: `0.0` to `1.0`
- Considers tokens whose cumulative probability adds up to `top_p`
- Lower = more focused/conservative, Higher = more diverse

**Example:**
- `top_p=0.1` → only considers the top 10% most likely tokens (very conservative)
- `top_p=0.9` → considers top 90% of tokens (more creative)

**Note:** Use either `temperature` OR `top_p`, not both (they control similar things).

---

## Max_tokens

The **maximum number of tokens** the model will generate in its response.

- Prevents responses from going on too long.
- Too low → response gets cut off mid-sentence.
- Too high → wastes tokens (and money on paid APIs).

**Note:** 1 token ≈ 4 characters for English text, but it varies.

---

## RAG (Retrieval-Augmented Generation)

A technique that combines **retrieval** (finding relevant documents) with **generation** (LLM creating answers).

**How it works:**
1. User asks a question
2. System retrieves relevant chunks from documents
3. System adds those chunks as context to the prompt
4. LLM generates answer using both its knowledge AND the retrieved context

**Why use RAG?**
- LLMs have training cutoff dates (don't know recent info)
- LLMs can't access your private documents
- Provides **grounded answers** (based on actual sources)
- Reduces hallucinations

**Analogy:** Like giving a student a textbook during an exam, instead of relying only on memory.

---

## Embedding

A **vector representation** of text (array of numbers).

- Similar texts → similar vectors (close in vector space)
- Enables **semantic search** (find meaning, not just keywords)
- Example: "machine learning" and "ML" have similar embeddings even though words are different

**Example:**
```
"machine learning" → [0.2, -0.5, 0.8, 0.1, ...] (vector of 384 numbers)
"deep learning"   → [0.3, -0.4, 0.7, 0.2, ...] (similar vector)
```

**For data engineers:** Like creating a hash, but for semantic similarity instead of exact matches.

---

## Chunking

Breaking long documents into **smaller pieces** (chunks) for better retrieval.

- Vector search works better on smaller, focused pieces
- Overlap between chunks prevents losing context at boundaries
- Typical chunk size: 256-1024 tokens

**Example:**
```
Long document (5000 chars)
↓
Chunk 1: chars 0-512
Chunk 2: chars 462-974 (overlap of 50 chars)
Chunk 3: chars 924-1436 (overlap of 50 chars)
```

**For data engineers:** Like partitioning a large table into smaller, manageable pieces.

---

## Vector Store

A database optimized for storing and searching **vectors** (embeddings).

- Stores: document text, embedding vectors, metadata
- Enables fast similarity search (find similar vectors)
- Examples: Chroma, FAISS, Pinecone, Weaviate

**For data engineers:** Like a NoSQL database, but optimized for vector similarity queries instead of key-value lookups.

---

## Retrieval

Finding **relevant document chunks** for a user's question.

**Process:**
1. Convert query to embedding (same model as documents)
2. Search vector store for most similar document embeddings
3. Return top-k most similar chunks

**Similarity calculation:**
- Usually **cosine similarity** (angle between vectors)
- Or **Euclidean distance** (straight-line distance)
- Lower distance / higher similarity = more relevant

**For data engineers:** Like a JOIN operation, but matching on semantic similarity instead of exact keys.

---

## Reranking

**Reordering** retrieved chunks using a more powerful model to improve relevance.

**Two-stage process:**
1. **Retrieve more** (e.g., top 10) using fast vector search
2. **Rerank** using slower but more accurate model (CrossEncoder)
3. **Take top-k** after reranking (e.g., top 3)

**Trade-off:**
- ✅ Better relevance
- ❌ Slower (reranking model is expensive)

**For data engineers:** Like a two-stage query: fast index scan, then detailed evaluation of candidates.

---

## Grounding

Ensuring that an answer is **based on** the retrieved context, not hallucinated.

- **Grounded answer**: Supported by retrieved documents
- **Ungrounded answer**: Made up or from LLM's training data (not from your documents)

**Why it matters:**
- RAG's main benefit is providing grounded answers
- If answers aren't grounded, RAG isn't working as intended

**How to check:**
- Simple: Check if key words from answer appear in retrieved chunks
- Advanced: Use LLM-as-judge to verify answer is supported by context

---

## Metadata Filtering

Filtering retrieved chunks by **additional information** (source, date, type, etc.).

**Example filters:**
- Only search in documents from 2024
- Only search in API documentation
- Exclude certain sources

**For data engineers:** Like adding WHERE conditions to a SELECT query, but for vector search.

---

*More terms will be added here as you progress through sections 04–07.*

