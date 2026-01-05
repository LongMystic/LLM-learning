"""
Basic RAG (Retrieval-Augmented Generation) implementation
This is the minimal version: chunk → embed → store → retrieve → generate
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import time

# Add parent directory to path to import from section 02
sys.path.append(str(Path(__file__).parent.parent / "02_llm_apis_and_prompts"))
from basic_api_calls import call_ollama


# ============================================
# Step 1: Load documents and chunk them
# ============================================

def chunk_text(text, chunk_size=512, chunk_overlap=50):
    """
    Simple text chunking by character count with overlap
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        
        # Stop if we've reached the end
        if end >= len(text):
            break
    
    return chunks


# ============================================
# Step 2: Create embeddings
# ============================================

def load_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    """
    Load a sentence transformer model for embeddings
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Model loaded!")
    return model


# ============================================
# Step 3: Store in vector database (Chroma)
# ============================================

def create_vector_store(collection_name="rag_docs"):
    """
    Create a Chroma vector store
    
    Args:
        collection_name: Name for the collection
    
    Returns:
        Chroma collection
    """
    # Create or get Chroma client
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"  # Persist to disk
    ))
    
    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Loaded existing collection: {collection_name}")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
    
    return collection


def add_documents_to_store(collection, documents, embeddings, ids=None, metadatas=None):
    """
    Add documents and their embeddings to the vector store
    
    Args:
        collection: Chroma collection
        documents: List of text chunks
        embeddings: List of embedding vectors
        ids: Optional list of IDs (auto-generated if None)
        metadatas: Optional list of metadata dicts
    """
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]
    
    if metadatas is None:
        metadatas = [{}] * len(documents)
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Added {len(documents)} documents to vector store")


# ============================================
# Step 4: Retrieve relevant chunks
# ============================================

def retrieve_chunks(collection, query_embedding, top_k=3):
    """
    Retrieve top-k most similar chunks for a query
    
    Args:
        collection: Chroma collection
        query_embedding: Embedding vector for the query
        top_k: Number of chunks to retrieve
    
    Returns:
        List of (document, distance, metadata) tuples
    """
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    # Format results
    retrieved = []
    for i in range(len(results['documents'][0])):
        retrieved.append({
            'text': results['documents'][0][i],
            'distance': results['distances'][0][i],
            'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
        })
    
    return retrieved


# ============================================
# Step 5: Generate answer with context
# ============================================

def generate_answer_with_rag(query, retrieved_chunks, model="llama3.2"):
    """
    Generate answer using retrieved context
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieved document chunks
        model: Ollama model name
    
    Returns:
        Generated answer
    """
    # Build context from retrieved chunks
    context = "\n\n".join([f"[Context {i+1}]: {chunk['text']}" 
                           for i, chunk in enumerate(retrieved_chunks)])
    
    # Create prompt with context
    prompt = f"""Use the following context to answer the question. 
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate answer
    answer = call_ollama(prompt=prompt, model=model, temperature=0.3, max_tokens=256)
    
    return answer


# ============================================
# Complete RAG pipeline
# ============================================

def build_rag_system(documents, embedding_model_name="BAAI/bge-small-en-v1.5"):
    """
    Build a complete RAG system from documents
    
    Args:
        documents: List of text documents
        embedding_model_name: Embedding model to use
    
    Returns:
        Tuple of (collection, embedding_model)
    """
    print("=" * 60)
    print("Building RAG System")
    print("=" * 60)
    
    # Load embedding model
    embedding_model = load_embedding_model(embedding_model_name)
    
    # Chunk documents
    print("\nChunking documents...")
    all_chunks = []
    all_metadatas = []
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc, chunk_size=512, chunk_overlap=50)
        all_chunks.extend(chunks)
        # Add metadata to track source document
        all_metadatas.extend([{"source": f"doc_{doc_idx}", "chunk_idx": i} 
                             for i in range(len(chunks))])
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Create embeddings
    print("\nCreating embeddings...")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
    print(f"Created {len(embeddings)} embeddings")
    
    # Create vector store
    print("\nCreating vector store...")
    collection = create_vector_store()
    
    # Add to vector store
    add_documents_to_store(collection, all_chunks, embeddings.tolist(), 
                          metadatas=all_metadatas)
    
    print("\n" + "=" * 60)
    print("RAG system built successfully!")
    print("=" * 60)
    
    return collection, embedding_model


def query_rag(collection, embedding_model, query, top_k=3, model="llama3.2"):
    """
    Query the RAG system
    
    Args:
        collection: Chroma collection
        embedding_model: SentenceTransformer model
        query: User's question
        top_k: Number of chunks to retrieve
        model: Ollama model name
    
    Returns:
        Tuple of (answer, retrieved_chunks)
    """
    # Create query embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Retrieve chunks
    retrieved_chunks = retrieve_chunks(collection, query_embedding, top_k=top_k)
    
    # Generate answer
    answer = generate_answer_with_rag(query, retrieved_chunks, model=model)
    
    return answer, retrieved_chunks


# ============================================
# Example usage
# ============================================

if __name__ == "__main__":
    # Sample documents (in real use, load from files)
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
        """
    ]
    
    # Build RAG system
    collection, embedding_model = build_rag_system(documents)
    
    # Query examples
    queries = [
        "What is machine learning?",
        "How do transformers work?",
        "What is the difference between machine learning and deep learning?"
    ]
    
    print("\n" + "=" * 60)
    print("Querying RAG System")
    print("=" * 60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        start_time = time.time()
        answer, retrieved = query_rag(collection, embedding_model, query, top_k=2)
        elapsed = time.time() - start_time
        
        print(f"Answer: {answer}")
        print(f"\nRetrieved {len(retrieved)} chunks:")
        for i, chunk in enumerate(retrieved):
            print(f"  [{i+1}] (distance: {chunk['distance']:.4f}) {chunk['text'][:100]}...")
        print(f"\nLatency: {elapsed:.2f} seconds")

