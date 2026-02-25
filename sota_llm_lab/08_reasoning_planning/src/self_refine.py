"""
08.2 – Self-reflection: LLM tự trả lời, tự review, rồi tự sửa.

Flow cho mỗi câu hỏi:
1) Draft lần 1
2) Critique: tự đánh giá + chỉ ra điểm thiếu/sai
3) Revised: sửa lại câu trả lời dựa trên critique
"""

from pathlib import Path
import sys
from typing import List, Dict

ROOT = Path(__file__).resolve().parent.parent.parent  # .../sota_llm_lab
GATEWAY_SRC = ROOT / "06_serving_llmops" / "src"
sys.path.append(str(GATEWAY_SRC))

from llm_gateway_client import chat_via_gateway  # type: ignore


QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "sr1",
        "question": "Giải thích Agentic AI là gì, liên hệ với các module bạn đã học (MCP, Agents, RAG, PEFT).",
    },
    {
        "id": "sr2",
        "question": "Tóm tắt lại pipeline RAG cơ bản trong project này.",
    },
]


def answer_draft(question: str, model: str = "llama3.2") -> str:
    msgs = [
        {"role": "system", "content": "Bạn trả lời ngắn gọn, rõ ràng, không cần quá hoàn hảo."},
        {"role": "user", "content": question},
    ]
    return chat_via_gateway(msgs, model=model, temperature=0.7, max_tokens=512)


def critique_answer(question: str, draft: str, model: str = "llama3.2") -> str:
    prompt = f"""
Bạn là critic.

Yêu cầu gốc:
\"\"\"{question}\"\"\"

Câu trả lời hiện tại:
\"\"\"{draft}\"\"\"

Nhiệm vụ:
1) Chỉ ra điểm thiếu sót / chưa chính xác / chưa rõ ràng.
2) Đề xuất cách cải thiện.
Trả lời ngắn gọn.
"""
    msgs = [
        {"role": "system", "content": "Bạn là critic, review câu trả lời và chỉ ra điểm cần cải thiện."},
        {"role": "user", "content": prompt},
    ]
    return chat_via_gateway(msgs, model=model, temperature=0.3, max_tokens=512)


def refine_answer(question: str, draft: str, critique: str, model: str = "llama3.2") -> str:
    prompt = f"""
Bạn là assistant.

Yêu cầu gốc:
\"\"\"{question}\"\"\"

Câu trả lời nháp:
\"\"\"{draft}\"\"\"

Nhận xét/critique:
\"\"\"{critique}\"\"\"

Hãy viết lại câu trả lời tốt hơn, sửa các điểm yếu đã nêu, nhưng vẫn ngắn gọn, rõ ràng.
"""
    msgs = [
        {"role": "system", "content": "Bạn sửa câu trả lời dựa trên critique, cho bản tốt hơn."},
        {"role": "user", "content": prompt},
    ]
    return chat_via_gateway(msgs, model=model, temperature=0.7, max_tokens=512)


def main():
    for q in QUESTIONS:
        qid = q["id"]
        question = q["question"]

        print("\n" + "=" * 80)
        print(f"ID: {qid}")
        print("Question:", question)

        draft = answer_draft(question)
        print("\n[DRAFT]")
        print(draft)

        critique = critique_answer(question, draft)
        print("\n[CRITIQUE]")
        print(critique)

        refined = refine_answer(question, draft, critique)
        print("\n[REFINED]")
        print(refined)
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()