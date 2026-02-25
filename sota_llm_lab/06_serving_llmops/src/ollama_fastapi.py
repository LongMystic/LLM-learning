"""
FastAPI LLM Gateway dùng Ollama (OpenAI-compatible).

Endpoint chính:
- POST /chat: nhận messages (giống OpenAI), trả kết quả từ Ollama.
- 07.4: log mỗi request vào logs/gateway_requests.jsonl.
"""
import json
import time
from pathlib import Path
from typing import List, Literal, Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ollama_client import chat_ollama

# 07.4 – Logging
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH = LOG_DIR / "gateway_requests.jsonl"


def log_request(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

Role = Literal["system", "user", "assistant"]

app = FastAPI(title="Ollama LLM Gateway")

class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    model: str = "llama3.2"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


@app.post("/chat")
def chat(req: ChatRequest):
    t0 = time.time()
    data = chat_ollama(
        messages=[m.dict() for m in req.messages],
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=req.stream,
    )
    elapsed = time.time() - t0

    prompt_preview = (req.messages[-1].content[:200] if req.messages else "") or ""
    answer_text = (data.get("message") or {}).get("content", "") if isinstance(data, dict) else str(data)
    answer_preview = (answer_text[:200]) if answer_text else ""

    log_request({
        "ts": time.time(),
        "model": req.model,
        "latency_sec": round(elapsed, 3),
        "prompt_preview": prompt_preview,
        "answer_preview": answer_preview,
    })

    return data


if __name__ == "__main__":
    uvicorn.run("ollama_fastapi:app", host="0.0.0.0", port=8000, reload=True)