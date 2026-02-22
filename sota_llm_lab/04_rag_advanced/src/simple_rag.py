"""
04.1 – RAG tối thiểu: chunking → embedding → vector search → Ollama answer.
Dùng sentence-transformers + ChromaDB (local, không cần server).
"""
import os
import json
import requests
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mcp_docs"
CHUNK_SIZE = 500        # ký tự tối đa mỗi chunk
CHUNK_OVERLAP = 50      # ký tự overlap giữa các chunk
TOP_K = 3               # số chunk trả về khi search


# ========= 1. Chunking =========

def load_md_files(docs_dir: str) -> list[dict]:
    """Đọc tất cả các file .md trong thư mục, trả về list {"path", "content"}."""
    print("LOADING MD FILES...")
    results = []
    for p in Path(docs_dir).rglob("*.md"):
        content = p.read_text(encoding="utf-8", errors="ignore")
        if content.strip():
            results.append({"path": str(p), "content": content})
    return results


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chia text thành các chunk nhỏ với overlap."""
    print("CHUNKING TEXT...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "text": chunk.strip(),
            "source": source,
            "start": start,
        })
        start = end - overlap
    return [c for c in chunks if c["text"]]


def chunk_all_docs(docs: list[dict]) -> list[dict]:
    """Chunk tất cả các docs."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["content"], source=doc["path"])
        all_chunks.extend(chunks)
    print(f"Tổng: {len(docs)} files -> {len(all_chunks)} chunks")
    return all_chunks


# ========= 2. Embedding =========

def build_index(chunks: list[dict], persist_dir: str = "./chroma_db") -> chromadb.Collection:
    """Tạo hoặc rebuild vector index từ chunks."""
    print("BUILDING INDEX...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=persist_dir)

    # Xóa collection cũ nếu có, tạo mới
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"], "start": c.get("start", 0)} for c in chunks]

    print("Đang embedding... (lần đầu có thể chậm)")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Đã index {len(chunks)} chunks vào ChromaDB tại {persist_dir}")
    return collection


def load_index(persist_dir: str = "./chroma_db") -> chromadb.Collection:
    """Load collection đã có."""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(name=COLLECTION_NAME)


# ========= 3. Vector Search =========

def search(query: str, collection: chromadb.Collection, top_k: int = TOP_K) -> list[dict]:
    """Tìm top_k chunks gần nhất với query."""
    print("SEARCHING...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_emb = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
    )

    hits = []
    for i in range(len(results['documents'][0])):
        hits.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return hits


# ========= 4. Answer with Ollama =========

def answer_with_context(query: str, hits: list[dict]) -> str:
    """Gọi Ollama với context từ RAG."""
    print("ANSWERING WITH CONTEXT...")
    context = "\n---\n".join(
        f"[Nguồn: {h['source']}]\n{h['text']}" for h in hits
    )

    prompt = f"""Dựa vào các đoạn tài liệu bên dưới, hãy trả lời câu hỏi của user.
CHỈ dùng thông tin từ tài liệu. Nếu không tìm thấy câu trả lời, nói rõ "Không tìm thấy trong tài liệu".

Tài liệu:
{context}

Câu hỏi: {query}

Trả lời:"""

    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ========= 5. Interactive =========

def interactive_rag(docs_dir: str, persist_dir: str = "./chroma_db"):
    # Bước 1: Load docs + chunk + index
    docs = load_md_files(docs_dir)
    if not docs:
        print(f"Không tìm thấy file .md nào trong {docs_dir}")
        return
    
    chunks = chunk_all_docs(docs)
    collection = build_index(chunks, persist_dir)

    # Bước 2: interactive Q&A
    print("\n=== RAG Q&A about MCP (gõ 'exit' để thoát) ===")
    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break

        hits = search(query, collection)
        print(f"\n[RAG] Tìm thấy {len(hits)} chunks liên quan:")
        for i, h in enumerate(hits, 1):
            print(f" {i}. [{h['source']}] {h['text'][:100]}... with distance: {h['distance']:.4f}")
        
        answer = answer_with_context(query, hits)
        print(f"\n[Answer]\n{answer}")


if __name__ == "__main__":
    import sys
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "sota_llm_lab/_01_mcp/docs"
    print(os.path.abspath(docs_dir))
    interactive_rag(docs_dir)