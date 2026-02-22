"""
04.3 – RAG tinh chỉnh: chunk theo mục (markdown), re-rank đơn giản, trả lời có trích nguồn.
Dùng chung embedding + ChromaDB với 04.1; thêm bước re-rank và format answer.
"""
import re
import requests
from pathlib import Path

from simple_rag import (
    load_md_files,
    build_index,
    load_index,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    OLLAMA_URL,
    MODEL_NAME,
    TOP_K,
)
from sentence_transformers import SentenceTransformer
import chromadb

OLLAMA_URL = OLLAMA_URL
MODEL_NAME = MODEL_NAME
PERSIST_DIR_V2 = "./chroma_db_v2"
TOP_K_RETRIEVAL = 8
TOP_K_AFTER_RERANK = 3
MAX_ANSWER_CONTEXT_CHARS = 4000

# ========= 1. Chunking theo mục (markdown headers) =========

def chunk_by_headers(content: str, source: str) -> list[dict]:
    """
    Chia theo ## hoăc ###, mỗi block gồm tiêu đề + nội dung đến mục kế.
    Nếu không có header thì fallback chunk 400 ký tự.
    """
    blocks = []
    pattern = r"^(#{1,3})\s+(.+)$"
    lines = content.split("\n")
    current_header = ""
    current_text = []

    for line in lines:
        m = re.match(pattern, line)
        if m:
            if current_text:
                text = "\n".join(current_text).strip()
                if text:
                    blocks.append({
                        "text": f"## {current_header}\n{text}" if current_header else text,
                        "source": source,
                    })
            current_header = m.group(2).strip()
            current_text = [line]
        else:
            current_text.append(line)

    if current_text:
        text = "\n".join(current_text).strip()
        if text:
            blocks.append({
                "text": f"## {current_header}\n{text}" if current_header else text,
                "source": source,
            })
    
    if not blocks:
        start = 0
        while start < len(content):
            chunk = content[start: start + 400].strip()
            if chunk:
                blocks.append({"text": chunk, "source": source})
            start += 400 - 50
    return blocks

def build_index_v2(docs_dir: str, persist_dir: str = PERSIST_DIR_V2) -> chromadb.Collection:
    docs = load_md_files(docs_dir)
    if not docs:
        raise FileNotFoundError(f"Không tìm thấy file .md nào trong {docs_dir}")
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_by_headers(d["content"], d["path"]))
    print(f"Chunk theo mục: {len(docs)} files -> {len(all_chunks)} chunks")
    return build_index(all_chunks, persist_dir)

# ========= 2. Re-rank đơn giản (keyword overlap) =========

def rerank_by_keywords(query: str, hits: list[dict], top_k: int = TOP_K_AFTER_RERANK) -> list[dict]:
    """
    Re-rank: ưu tiên chunk có nhiều từ khóa của query xuất hiện (case-insensitive).
    """
    q_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for h in hits:
        text = (h.get("text") or "").lower()
        overlap = sum(1 for w in q_words if w in text)
        scored.append((overlap, h))
    scored.sort(key=lambda x: -x[0])
    return [h for _, h in scored[:top_k]]

def search_and_rerank(
    query: str,
    collection: chromadb.Collection,
    embedder: SentenceTransformer,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_AFTER_RERANK,
) -> list[dict]:
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k_retrieval)
    hits = []
    for i in range(len(results['documents'][0])):
        hits.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return rerank_by_keywords(query, hits, top_k=top_k_final)


# ========= 3. Format answer với trích nguồn =========

def answer_with_sources(query: str, hits: list[dict]) -> tuple[str, list[str]]:
    """
    Gọi Ollama với context, yêu cầu trả lời và liệt kê nguồn (file) đã dùng.
    Trả về (answer_text, list_source_paths).
    """
    context = "\n---\n".join(
        f"[Nguồn: {h['source']}]\n{h['text']}" for h in hits
    )
    if len(context) > MAX_ANSWER_CONTEXT_CHARS:
        context = context[:MAX_ANSWER_CONTEXT_CHARS] + "\n...[đã cắt bớt]"

    prompt = f"""Dựa vào các đoạn tài liệu dưới đây, trả lời câu hỏi của user.
- CHỈ dùng thông tin trong tài liệu.
- Cuối trả lời, ghi rõ: "Nguồn: [liệt kê đường dẫn file đã dùng]"

Tài liệu:
{context}

Câu hỏi: {query}

Trả lời:"""

    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:500]}")
    answer = resp.json().get("response", "").strip()

    sources = list({h["source"] for h in hits})
    return answer, sources


# ---------- 4. Interactive ----------

def interactive_rag_v2(docs_dir: str, persist_dir: str = PERSIST_DIR_V2):
    from pathlib import Path
    index_path = Path(persist_dir)
    need_build = True
    if index_path.exists():
        try:
            collection = load_index(persist_dir)
            try:
                n = collection.count()
            except AttributeError:
                n = len(collection.get(limit=1)["ids"]) if collection.get(limit=1)["ids"] else 0
            if n > 0:
                print(f"Load index từ {persist_dir} ({n} chunks)")
                need_build = False
            else:
                print("Index tồn tại nhưng rỗng, sẽ build lại.")
        except Exception as e:
            print(f"Không load được index: {e}, sẽ build lại.")
    if need_build:
        print(f"Build index v2 (chunk theo mục) từ {docs_dir}")
        collection = build_index_v2(docs_dir, persist_dir)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("\n=== RAG v2 (chunk theo mục + re-rank + nguồn) – gõ 'exit' để thoát ===")
    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        try:
            hits = search_and_rerank(q, collection, embedder)
            if not hits:
                print("Không tìm thấy chunk liên quan.")
                continue
            print(f"[RAG] Đã re-rank, dùng {len(hits)} chunks.")
            answer, sources = answer_with_sources(q, hits)
            print("\n[Answer]\n", answer)
            print("\n[Nguồn]", sources)
        except Exception as e:
            print(f"[Lỗi] {e}")


if __name__ == "__main__":
    import sys
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "../../_01_mcp/docs"
    interactive_rag_v2(docs_dir)