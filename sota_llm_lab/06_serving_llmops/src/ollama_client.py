"""
Simple Ollama client wrapper cho các module khác dùng chung.

- Dùng HTTP API của Ollama (mặc định: http://localhost:11434).
- Cung cấp hàm chat_ollama(messages, model=..., stream=False).
"""
from typing import List, Dict, Any, Literal, Optional
import os
import requests

OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/v1/chat/completions"

Role = Literal["system", "user", "assistant"]

def chat_ollama(
    messages: List[Dict[str, Any]],
    model: str = "llama3.2",
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Gọi Ollama theo format gần giống OpenAI /v1/chat/completions.

    messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    Trả về: dict JSON từ Ollama (hoặc iterator nếu stream=True sau này).
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def simple_chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "llama3.2",
    **kwargs: Any,
) -> str:
    """
    Hàm tiện dụng: truyền 1 prompt (và optional system_prompt), trả về text trả lời.
    """
    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})

    data = chat_ollama(msgs, model, **kwargs)

    # Ollama theo OpenAI-compatible: choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return str(data)


if __name__ == "__main__":
    # Test nhanh: yêu cầu Ollama giải thích LoRA
    print("=== Ollama Client Test ===")
    user_prompt = "Giải thích ngắn gọn: LoRA trong fine-tune LLM là gì?"
    system_prompt = "Bạn là trợ lý kỹ thuật, trả lời ngắn gọn, rõ ràng."
    answer = simple_chat(
        user_prompt,
        system_prompt,
        model="llama3.2",
        max_tokens=256,
    )
    print("Answer:\n", answer)