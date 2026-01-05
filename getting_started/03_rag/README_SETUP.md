# Setup Instructions for Section 03 (RAG)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Make sure Ollama is running:**
   ```bash
   ollama serve
   ```
   (You should have this set up from section 02)

3. **Pull a model if you haven't:**
   ```bash
   ollama pull llama3.2
   ```

4. **Run the examples:**
   ```bash
   # Basic RAG
   python basic_rag.py
   
   # RAG with reranking
   python rag_with_reranking.py
   
   # RAG with metadata filtering
   python rag_with_metadata_filtering.py
   
   # Evaluate RAG system
   python evaluate_rag.py
   ```

## File Structure

- `basic_rag.py` - Minimal RAG implementation (chunk → embed → store → retrieve → generate)
- `rag_with_reranking.py` - Adds reranking for better retrieval quality
- `rag_with_metadata_filtering.py` - Adds metadata filtering (source, date, type)
- `evaluate_rag.py` - Evaluation script with metrics (grounding, F1, latency)
- `explain_about_code.md` - Detailed explanations for beginners
- `README_SETUP.md` - This file

## First-Time Setup Notes

### Embedding Models

The code uses `BAAI/bge-small-en-v1.5` by default. On first run, it will:
- Download the model (about 130 MB)
- Cache it locally (won't download again)

**Alternative models:**
- `BAAI/bge-base-en-v1.5` - Larger, better quality (400 MB)
- `BAAI/bge-large-en-v1.5` - Best quality, but slow (1.3 GB)
- `sentence-transformers/all-MiniLM-L6-v2` - Very small, fast (80 MB)

To use a different model, change the `model_name` parameter in the code.

### Vector Database (Chroma)

Chroma will create a local database in `./chroma_db/` directory:
- Persists between runs (you don't need to rebuild every time)
- Can delete the folder to start fresh
- No separate database server needed (runs locally)

### Reranking Model

`rag_with_reranking.py` uses `BAAI/bge-reranker-base`:
- Downloads on first use (about 420 MB)
- Slower than embedding models (runs on CPU by default)
- Significantly improves retrieval quality

## Troubleshooting

### "Model not found" error

**Problem:** SentenceTransformer can't find the model.

**Solution:**
```bash
# Make sure you have internet connection for first download
# Or manually download:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### "Chroma connection error"

**Problem:** Chroma can't create/write to database directory.

**Solution:**
- Make sure you have write permissions in the current directory
- Or specify a different `persist_directory` in the code

### "Ollama connection error"

**Problem:** Can't connect to Ollama.

**Solution:**
- Make sure Ollama is running: `ollama serve`
- Check if it's on the default port: http://localhost:11434
- Try: `ollama list` to verify Ollama is working

### Slow embedding generation

**Problem:** Creating embeddings takes a long time.

**Solutions:**
- Use smaller embedding model (`bge-small` instead of `bge-large`)
- Use GPU if available (PyTorch will auto-detect)
- Process in batches (code already does this)

### Out of memory errors

**Problem:** Running out of RAM when processing large documents.

**Solutions:**
- Use smaller embedding model
- Process documents in smaller batches
- Reduce chunk size
- Use CPU instead of GPU (if GPU memory is the issue)

## Testing Your Setup

Run this quick test:

```python
from basic_rag import build_rag_system, query_rag

# Simple test
documents = ["Machine learning is a subset of AI."]
collection, embedding_model = build_rag_system(documents)
answer, chunks = query_rag(collection, embedding_model, "What is machine learning?")
print(answer)
```

If this works, your setup is correct!

## Next Steps

1. **Start with `basic_rag.py`** - Understand the core pipeline
2. **Read `explain_about_code.md`** - Deep dive into how it works
3. **Try `rag_with_reranking.py`** - See how reranking improves quality
4. **Experiment with `rag_with_metadata_filtering.py`** - Learn about filtering
5. **Run `evaluate_rag.py`** - Measure your system's performance

Then you'll be ready to build your own RAG system!

