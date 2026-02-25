"""
09.1 – Pseudo-MoE router đơn giản (rule-based).

- Query về code/kỹ thuật → model A (vd: "llama3.2").
- Query về viết lách/email → model B (vd: "qwen2.5" hoặc model khác bạn có).
"""

from pathlib import Path
import sys
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent.parent  # .../sota_llm_lab
GATEWAY_SRC = ROOT / "06_serving_llmops" / "src"
sys.path.append(str(GATEWAY_SRC))

from llm_gateway_client import chat_via_gateway  # type: ignore


def choose_model(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["code", "python", "bug", "function", "class", "api"]):
        return "llama3.2"  # model A: technical
    if any(k in q for k in ["email", "thư", "viết", "blog", "story"]):
        return "qwen2.5"  # model B: writing (đổi theo model bạn có)
    return "llama3.2"


def moe_chat(user_query: str) -> Dict[str, str]:
    model = choose_model(user_query)
    msgs = [
        {"role": "system", "content": f"Bạn là trợ lý dùng model {model}."},
        {"role": "user", "content": user_query},
    ]
    answer = chat_via_gateway(msgs, model=model, temperature=0.7, max_tokens=512)
    return {"model": model, "answer": answer}


def main():
    print("=== Pseudo-MoE Router (rule-based) – gõ 'exit' để thoát ===")
    while True:
        q = input("\nUser> ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        result = moe_chat(q)
        print(f"[Chosen model] {result['model']}")
        print("[Answer]")
        print(result["answer"])


if __name__ == "__main__":
    main()

