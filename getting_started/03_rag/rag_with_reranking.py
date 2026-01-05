"""
RAG with reranking - improves retrieval quality
Reranking uses a more powerful model to reorder retrieved chunks
"""

import sys
from pathlib import Path
from sentence_transformers import CrossEncoder
import time

# Add current directory to path to import from basic_rag
sys.path.append(str(Path(__file__).parent))
from basic_rag import (
    build_rag_system, query_rag, retrieve_chunks,
    load_embedding_model, create_vector_store
)


# ============================================
# Reranking model
# ============================================

def load_reranker(model_name="BAAI/bge-reranker-base"):
    """
    Load a reranking model (CrossEncoder)
    
    Args:
        model_name: HuggingFace model name for reranking
    
    Returns:
        CrossEncoder model
    """
    print(f"Loading reranker: {model_name}")
    model = CrossEncoder(model_name)
    print("Reranker loaded!")
    return model


def rerank_chunks(reranker, query, chunks, top_k=None):
    """
    Rerank retrieved chunks using a reranker model
    
    Args:
        reranker: CrossEncoder model
        query: User's question
        chunks: List of retrieved chunks (dicts with 'text' key)
        top_k: Number of top chunks to return (None = return all)
    
    Returns:
        List of reranked chunks (sorted by relevance score)
    """
    # Prepare pairs for reranking: (query, chunk_text)
    pairs = [[query, chunk['text']] for chunk in chunks]
    
    # Get relevance scores
    scores = reranker.predict(pairs)
    
    # Add scores to chunks and sort
    for i, chunk in enumerate(chunks):
        chunk['rerank_score'] = float(scores[i])
    
    # Sort by rerank score (higher = more relevant)
    reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
    
    # Return top_k if specified
    if top_k is not None:
        return reranked[:top_k]
    
    return reranked


# ============================================
# RAG with reranking pipeline
# ============================================

def query_rag_with_reranking(collection, embedding_model, reranker, query, 
                             initial_top_k=10, final_top_k=3, model="llama3.2"):
    """
    Query RAG system with reranking
    
    Process:
    1. Retrieve more chunks initially (initial_top_k)
    2. Rerank them using the reranker model
    3. Take top final_top_k after reranking
    4. Generate answer
    
    Args:
        collection: Chroma collection
        embedding_model: SentenceTransformer model
        reranker: CrossEncoder reranker model
        query: User's question
        initial_top_k: How many chunks to retrieve initially
        final_top_k: How many chunks to use after reranking
        model: Ollama model name
    
    Returns:
        Tuple of (answer, reranked_chunks)
    """
    # Step 1: Create query embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Step 2: Retrieve more chunks than needed (for reranking)
    retrieved_chunks = retrieve_chunks(collection, query_embedding, top_k=initial_top_k)
    
    # Step 3: Rerank chunks
    reranked_chunks = rerank_chunks(reranker, query, retrieved_chunks, top_k=final_top_k)
    
    # Step 4: Generate answer with top reranked chunks
    from basic_rag import generate_answer_with_rag
    answer = generate_answer_with_rag(query, reranked_chunks, model=model)
    
    return answer, reranked_chunks


# ============================================
# Comparison: RAG vs RAG with reranking
# ============================================

def compare_rag_methods(collection, embedding_model, query, model="llama3.2"):
    """
    Compare basic RAG vs RAG with reranking
    """
    print("=" * 60)
    print("Comparing RAG Methods")
    print("=" * 60)
    print(f"\nQuery: {query}\n")
    
    # Basic RAG
    print("--- Basic RAG (no reranking) ---")
    start_time = time.time()
    answer_basic, chunks_basic = query_rag(collection, embedding_model, query, top_k=3, model=model)
    time_basic = time.time() - start_time
    
    print(f"Answer: {answer_basic}")
    print(f"Top chunks:")
    for i, chunk in enumerate(chunks_basic):
        print(f"  [{i+1}] (distance: {chunk['distance']:.4f}) {chunk['text'][:80]}...")
    print(f"Time: {time_basic:.2f}s\n")
    
    # RAG with reranking
    print("--- RAG with Reranking ---")
    reranker = load_reranker()
    
    start_time = time.time()
    answer_rerank, chunks_rerank = query_rag_with_reranking(
        collection, embedding_model, reranker, query,
        initial_top_k=10, final_top_k=3, model=model
    )
    time_rerank = time.time() - start_time
    
    print(f"Answer: {answer_rerank}")
    print(f"Top chunks (after reranking):")
    for i, chunk in enumerate(chunks_rerank):
        print(f"  [{i+1}] (rerank_score: {chunk['rerank_score']:.4f}, "
              f"original_distance: {chunk['distance']:.4f}) {chunk['text'][:80]}...")
    print(f"Time: {time_rerank:.2f}s\n")
    
    print("=" * 60)
    print("Summary:")
    print(f"  Basic RAG time: {time_basic:.2f}s")
    print(f"  Reranked RAG time: {time_rerank:.2f}s")
    print(f"  Time overhead: {time_rerank - time_basic:.2f}s")
    print("=" * 60)


# ============================================
# Example usage
# ============================================

if __name__ == "__main__":
    # Sample documents
    documents = [
        """
        Machine learning is a subset of artificial intelligence that enables computers 
        to learn from data without being explicitly programmed. It uses algorithms to 
        identify patterns in data and make predictions or decisions.
        """,
        """
        Deep learning is a type of machine learning that uses neural networks with 
        multiple layers. These networks can learn complex patterns in data, such as 
        recognizing images, understanding speech, or translating languages.
        """,
        """
        Transformers are a type of neural network architecture introduced in 2017. 
        They use attention mechanisms to process sequences of data, making them 
        particularly effective for natural language processing tasks.
        """,
        """
        Natural language processing (NLP) is a field of AI that focuses on enabling 
        computers to understand, interpret, and generate human language. Applications 
        include chatbots, translation, and sentiment analysis.
        """
    ]
    
    # Build RAG system (reuse if already built)
    print("Building RAG system...")
    collection, embedding_model = build_rag_system(documents)
    
    # Compare methods
    test_query = "What is the relationship between transformers and natural language processing?"
    compare_rag_methods(collection, embedding_model, test_query)

