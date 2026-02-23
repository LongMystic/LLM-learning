"""
FastAPI LLM Gateway dùng Ollama (OpenAI-compatible).

Endpoint chính:
- POST /chat: nhận messages (giống OpenAI), trả kết quả từ Ollama.
"""
from typing import List, Literal, Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ollama_client import chat_ollama

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
    data = chat_ollama(
        messages=[m.dict() for m in req.messages],
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=req.stream,
    )
    return data


if __name__ == "__main__":
    uvicorn.run("ollama_fastapi:app", host="0.0.0.0", port=8000, reload=True)