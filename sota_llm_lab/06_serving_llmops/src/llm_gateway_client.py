"""
Client Python gọi LLM Gateway (FastAPI) thay vì gọi Ollama trực tiếp.

Giả sử gateway đang chạy ở: http://localhost:8000
"""

from typing import List, Dict, Literal, Any
import os
import requests

Role = Literal["system", "user", "assistant"]

GATEWAY_URL = 'http://localhost:8000/chat'


def chat_via_gateway(
    messages: List[Dict[str, Any]],
    model: str = "llama3.2",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    resp = requests.post(GATEWAY_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Nếu gateway trả raw từ Ollama native:
    # data["message"]["content"]
    try:
        return data["message"]["content"]
    except Exception:
        return str(data)


if __name__ == "__main__":
    msgs = [
        {"role": "system", "content": "Bạn là trợ lý kỹ thuật, trả lời ngắn gọn."},
        {"role": "user", "content": "Orchestrator trong Agentic AI là gì?"},
    ]
    print(chat_via_gateway(msgs))