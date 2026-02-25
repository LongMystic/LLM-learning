"""
09.2 – LLM-as-router: dùng 1 model router để chọn expert LLM.

- Router model: 1 LLM (vd: "llama3.2") phân loại intent.
- Expert models: model A/B/C khác nhau tuỳ loại câu hỏi.
"""

from pathlib import Path
import sys
from typing import Dict, Literal
import json

ROOT = Path(__file__).resolve().parent.parent.parent  # .../sota_llm_lab
GATEWAY_SRC = ROOT / "06_serving_llmops" / "src"
sys.path.append(str(GATEWAY_SRC))

from llm_gateway_client import chat_via_gateway  # type: ignore

Intent = Literal["code", "writing", "rag", "other"]


def route_intent(query: str, router_model: str = "llama3.2") -> Intent:
    prompt = f"""
Phân loại câu hỏi sau vào một trong các nhóm: "code", "writing", "rag", "other".

Câu hỏi:
\"\"\"{query}\"\"\"

Trả về DUY NHẤT một JSON:
{{ "intent": "code" }}  hoặc  {{ "intent": "writing" }}  hoặc  {{ "intent": "rag" }}  hoặc  {{ "intent": "other" }}.
Không thêm bất kỳ text nào khác.
"""
    msgs = [
        {"role": "system", "content": "Bạn là router, chỉ trả về JSON intent như yêu cầu."},
        {"role": "user", "content": prompt},
    ]
    text = chat_via_gateway(msgs, model=router_model, temperature=0.0, max_tokens=256).strip()
    try:
        data = json.loads(text)
        intent = data.get("intent", "other")
        if intent not in ("code", "writing", "rag", "other"):
            return "other"
        return intent  # type: ignore[return-value]
    except Exception:
        return "other"


def choose_model_by_intent(intent: Intent) -> str:
    if intent == "code":
        return "llama3.2"
    if intent == "writing":
        return "qwen2.5"
    if intent == "rag":
        return "llama3.2"  # hoặc model chuyên tóm tắt docs/RAG
    return "llama3.2"


def moe_chat_llm_router(user_query: str) -> Dict[str, str]:
    intent = route_intent(user_query)
    model = choose_model_by_intent(intent)
    msgs = [
        {"role": "system", "content": f"Bạn là trợ lý (intent={intent}), dùng model {model}."},
        {"role": "user", "content": user_query},
    ]
    answer = chat_via_gateway(msgs, model=model, temperature=0.7, max_tokens=512)
    return {"intent": intent, "model": model, "answer": answer}


def main():
    print("=== LLM-as-router MoE – gõ 'exit' để thoát ===")
    while True:
        q = input("\nUser> ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        result = moe_chat_llm_router(q)
        print(f"[Intent] {result['intent']}  |  [Model] {result['model']}")
        print("[Answer]")
        print(result["answer"])


if __name__ == "__main__":
    main()

