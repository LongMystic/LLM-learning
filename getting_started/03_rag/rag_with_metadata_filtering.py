"""
RAG with metadata filtering
Filter retrieved chunks by metadata (source, date, type, etc.)
"""

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime
import time

# Add current directory to path to import from basic_rag
sys.path.append(str(Path(__file__).parent))
from basic_rag import (
    build_rag_system, retrieve_chunks,
    load_embedding_model, create_vector_store
)


# ============================================
# Enhanced document storage with metadata
# ============================================

def add_documents_with_metadata(collection, documents, embeddings, 
                                sources, dates=None, doc_types=None):
    """
    Add documents with rich metadata
    
    Args:
        collection: Chroma collection
        documents: List of text chunks
        embeddings: List of embedding vectors
        sources: List of source identifiers
        dates: List of dates (strings or datetime objects)
        doc_types: List of document types (e.g., "article", "tutorial", "api_doc")
    """
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Build metadata
    metadatas = []
    for i in range(len(documents)):
        metadata = {"source": sources[i]}
        
        if dates:
            if isinstance(dates[i], datetime):
                metadata["date"] = dates[i].isoformat()
            else:
                metadata["date"] = dates[i]
        
        if doc_types:
            metadata["type"] = doc_types[i]
        
        metadatas.append(metadata)
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Added {len(documents)} documents with metadata")


# ============================================
# Metadata filtering
# ============================================

def retrieve_with_filters(collection, query_embedding, top_k=10,
                          source_filter=None, date_filter=None, type_filter=None):
    """
    Retrieve chunks with metadata filtering
    
    Args:
        collection: Chroma collection
        query_embedding: Query embedding vector
        top_k: Number of chunks to retrieve
        source_filter: Filter by source (exact match or list)
        date_filter: Filter by date (dict with 'gte', 'lte', etc.)
        type_filter: Filter by document type (exact match or list)
    
    Returns:
        List of filtered chunks
    """
    # Build where clause for Chroma
    where_clause = {}
    
    if source_filter:
        if isinstance(source_filter, list):
            where_clause["source"] = {"$in": source_filter}
        else:
            where_clause["source"] = source_filter
    
    if type_filter:
        if isinstance(type_filter, list):
            where_clause["type"] = {"$in": type_filter}
        else:
            where_clause["type"] = type_filter
    
    # Note: Chroma's date filtering is limited, so we filter after retrieval
    # For production, consider using a more advanced vector DB
    
    # Query with filters
    if where_clause:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
    
    # Format results
    retrieved = []
    for i in range(len(results['documents'][0])):
        chunk = {
            'text': results['documents'][0][i],
            'distance': results['distances'][0][i],
            'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
        }
        
        # Apply date filter if specified
        if date_filter and 'date' in chunk['metadata']:
            chunk_date = datetime.fromisoformat(chunk['metadata']['date'])
            
            if 'gte' in date_filter and chunk_date < date_filter['gte']:
                continue
            if 'lte' in date_filter and chunk_date > date_filter['lte']:
                continue
        
        retrieved.append(chunk)
    
    return retrieved


# ============================================
# Example: Building RAG with metadata
# ============================================

def build_rag_with_metadata(documents, sources, dates=None, doc_types=None,
                           embedding_model_name="BAAI/bge-small-en-v1.5"):
    """
    Build RAG system with metadata
    
    Args:
        documents: List of text documents
        sources: List of source identifiers
        dates: Optional list of dates
        doc_types: Optional list of document types
        embedding_model_name: Embedding model name
    
    Returns:
        Tuple of (collection, embedding_model)
    """
    print("=" * 60)
    print("Building RAG System with Metadata")
    print("=" * 60)
    
    # Load embedding model
    embedding_model = load_embedding_model(embedding_model_name)
    
    # Chunk documents
    print("\nChunking documents...")
    from basic_rag import chunk_text
    
    all_chunks = []
    all_sources = []
    all_dates = []
    all_types = []
    
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc, chunk_size=512, chunk_overlap=50)
        all_chunks.extend(chunks)
        all_sources.extend([sources[doc_idx]] * len(chunks))
        
        if dates:
            all_dates.extend([dates[doc_idx]] * len(chunks))
        if doc_types:
            all_types.extend([doc_types[doc_idx]] * len(chunks))
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Create embeddings
    print("\nCreating embeddings...")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
    
    # Create vector store
    print("\nCreating vector store...")
    collection = create_vector_store(collection_name="rag_docs_metadata")
    
    # Add with metadata
    add_documents_with_metadata(
        collection, all_chunks, embeddings.tolist(),
        all_sources, all_dates, all_types
    )
    
    print("\n" + "=" * 60)
    print("RAG system with metadata built successfully!")
    print("=" * 60)
    
    return collection, embedding_model


# ============================================
# Example usage
# ============================================

if __name__ == "__main__":
    # Sample documents with metadata
    documents = [
        "Machine learning enables computers to learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
        "Transformers use attention mechanisms for natural language processing tasks.",
        "Natural language processing focuses on enabling computers to understand human language.",
    ]
    
    sources = ["ml_basics.txt", "deep_learning.txt", "transformers.txt", "nlp_intro.txt"]
    dates = [
        datetime(2023, 1, 15),
        datetime(2023, 6, 20),
        datetime(2024, 1, 10),
        datetime(2024, 3, 5),
    ]
    doc_types = ["tutorial", "article", "research", "tutorial"]
    
    # Build RAG with metadata
    collection, embedding_model = build_rag_with_metadata(
        documents, sources, dates, doc_types
    )
    
    # Test queries with different filters
    query = "What is machine learning?"
    query_embedding = embedding_model.encode([query])[0]
    
    print("\n" + "=" * 60)
    print("Testing Metadata Filters")
    print("=" * 60)
    
    # No filter
    print("\n1. No filter (all documents):")
    chunks = retrieve_with_filters(collection, query_embedding, top_k=3)
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] Source: {chunk['metadata'].get('source')}, "
              f"Type: {chunk['metadata'].get('type')}, "
              f"Date: {chunk['metadata'].get('date')}")
    
    # Filter by source
    print("\n2. Filter by source (only ml_basics.txt):")
    chunks = retrieve_with_filters(collection, query_embedding, top_k=3,
                                   source_filter="ml_basics.txt")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] Source: {chunk['metadata'].get('source')}")
    
    # Filter by type
    print("\n3. Filter by type (only tutorials):")
    chunks = retrieve_with_filters(collection, query_embedding, top_k=3,
                                   type_filter="tutorial")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] Type: {chunk['metadata'].get('type')}, "
              f"Source: {chunk['metadata'].get('source')}")
    
    # Filter by date
    print("\n4. Filter by date (only 2024 documents):")
    chunks = retrieve_with_filters(collection, query_embedding, top_k=3,
                                   date_filter={'gte': datetime(2024, 1, 1)})
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] Date: {chunk['metadata'].get('date')}, "
              f"Source: {chunk['metadata'].get('source')}")

