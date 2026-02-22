"""
04.2 – Research Agent dùng RAG (index từ 04.1) → Writer Agent.
Chạy từ: sota_llm_lab/04_rag_advanced/src/
"""
import sys
from pathlib import Path

# thêm thư mục chứa simple_rag
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb
import requests
from simple_rag import (
    OLLAMA_URL,
    load_md_files,
    chunk_text,
    chunk_all_docs,
    build_index,
    load_index,
    search,
    answer_with_context,
    interactive_rag,
    CHUNK_SIZE,
    TOP_K
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
PERSIST_DIR = "./chroma_db"
DOCS_DIR_DEFAULT = "sota_llm_lab/_01_mcp/docs"

MAX_PROMPT_CHARS = 6000  # giới hạn prompt để tránh 500 do quá dài

def call_ollama(prompt: str, format_json: bool = False):
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n...[đã cắt bớt]"
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    if format_json:
        payload["format"] = "json"
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    if resp.status_code != 200:
        body = (resp.text or "(empty)")[:800]
        print(f"[DEBUG] Ollama {resp.status_code} | URL: {resp.url}")
        print(f"[DEBUG] Response body: {body}")
        raise RuntimeError(f"Ollama {resp.status_code}: {body}")
    raw = resp.json().get("response", "").strip()
    if format_json:
        import json
        return json.loads(raw)
    return raw

def research_agent_rag(user_goal: str, collection: chromadb.Collection) -> str:
    """
    Research agent chỉ dùng RAG: search → lấy top_k chunks → tóm tắt bằng LLM.
    """
    hits = search(user_goal, collection, top_k=TOP_K)
    if not hits:
        return "(RAG không tìm thấy dữ liệu)"

    max_ctx = 3500
    parts = []
    for h in hits:
        parts.append(f"{h['source']}\n{h['text']}")
        if sum(len(p) for p in parts) >= max_ctx:
            break
    context = "\n---\n".join(parts)

    prompt = f"""Dựa trên các đoạn tài liệu dưới đây, hãy tóm tắt ngắn gọn thông tin liên quan đến yêu cầu của user.
KHÔNG bịa thêm. Chỉ dùng nội dung trong tài liệu.

Yêu cầu user: {user_goal}

Tài liệu:
{context}

Tóm tắt:"""
    return call_ollama(prompt, format_json=False)


def writer_agent(research_summary: str, user_goal: str) -> str:
    if not research_summary or research_summary.startswith("(RAG không"):
        return "Không có dữ liệu để viết báo cáo. Research agent không tìm được thông tin."

    prompt = f"""Bạn là writer agent. Viết báo cáo ngắn gọn dựa trên dữ liệu bên dưới.
KHÔNG bịa thêm thông tin. 

Yêu cầu user: {user_goal}

Dữ liệu từ research (RAG):
{research_summary}

Báo cáo:"""
    return call_ollama(prompt, format_json=False)


def orchestrator_rag(user_goal: str, collection: chromadb.Collection) -> str:
    print(f"[Orchestrator] Yêu cầu: {user_goal}")
    print("[Orchestrator] Research (RAG)...")
    research_result = research_agent_rag(user_goal, collection)
    print(f"[Research/RAG] Tóm tắt (preview): {research_result[:400]}...")
    print("[Orchestrator] Writer...")
    report = writer_agent(research_result, user_goal)
    return report


def main():
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else DOCS_DIR_DEFAULT
    persist_dir = sys.argv[2] if len(sys.argv) > 2 else PERSIST_DIR

    # Load hoặc build index
    index_path = Path(persist_dir)
    if index_path.exists() and (index_path / "chroma_db.sqlite3").exists():
        print(f"Loading existing index from {persist_dir}")
        collection = load_index(persist_dir)
    else:
        print(f"Build index từ {docs_dir}...")
        docs = load_md_files(docs_dir)
        if not docs:
            print(f"Không tìm thấy file .md nào trong {docs_dir}")
            return
        chunks = chunk_all_docs(docs)
        collection = build_index(chunks, persist_dir)
    
    print("\n=== RAG + Research/Writer Agent (gõ 'exit' để thoát) ===")
    while True:
        goal = input("\nUser goal> ").strip()
        if goal.lower() in ("exit", "quit", "q"):
            break
        try:
            report = orchestrator_rag(goal, collection)
            print("\n" + "=" * 30)
            print("[Final Report]")
            print("=" * 30)
            print(report)
            print("=" * 30)
        except Exception as e:
            print(f"[Lỗi] {e}")


if __name__ == "__main__":
    main()