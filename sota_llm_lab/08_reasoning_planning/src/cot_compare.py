"""
08.1 – So sánh trả lời thường vs Chain-of-Thought (CoT) cho vài bài toán đơn giản.

Yêu cầu:
- Đã có LLM Gateway chạy (module 06).
- Đã có llm_gateway_client.chat_via_gateway trong PYTHONPATH (tương tự 07_x).
"""

from pathlib import Path
import sys
from typing import List, Dict

# Thêm đường dẫn tới 06_serving_llmops/src để import client
ROOT = Path(__file__).resolve().parent.parent.parent  # .../sota_llm_lab
GATEWAY_SRC = ROOT / "06_serving_llmops" / "src"
sys.path.append(str(GATEWAY_SRC))

from llm_gateway_client import chat_via_gateway  # type: ignore


TASKS: List[Dict[str, str]] = [
    {
        "id": "math1",
        "question": "Một cửa hàng bán 3 chiếc áo, mỗi chiếc 120k, và 2 chiếc quần, mỗi chiếc 200k. Tổng tiền là bao nhiêu?",
    },
    {
        "id": "logic1",
        "question": "Nếu hôm nay là thứ Hai, 10 ngày nữa là thứ mấy?",
    },
    {
        "id": "explain1",
        "question": "Giải thích ngắn gọn tại sao CoT (Chain-of-Thought) có thể giúp mô hình trả lời chính xác hơn.",
    },
]


def ask_no_cot(question: str, model: str = "llama3.2") -> str:
    msgs = [
        {"role": "system", "content": "Bạn trả lời trực tiếp, ngắn gọn, chỉ cho kết quả cuối cùng."},
        {"role": "user", "content": question},
    ]
    return chat_via_gateway(msgs, model=model, temperature=0.3, max_tokens=256)


def ask_with_cot(question: str, model: str = "llama3.2") -> str:
    prompt = (
        "Hãy giải bài toán sau từng bước một, ghi rõ lập luận.\n"
        "Sau khi suy luận xong, ở dòng cuối cùng hãy viết: 'Kết luận: ...' với câu trả lời cuối cùng.\n\n"
        f"Câu hỏi: {question}"
    )
    msgs = [
        {"role": "system", "content": "Bạn là trợ lý giải thích cẩn thận, show reasoning."},
        {"role": "user", "content": prompt},
    ]
    return chat_via_gateway(msgs, model=model, temperature=0.3, max_tokens=512)


def main():
    for task in TASKS:
        qid = task["id"]
        q = task["question"]

        print("\n" + "=" * 80)
        print(f"ID: {qid}")
        print("Question:", q)

        print("\n[NO CoT]")
        ans_no_cot = ask_no_cot(q)
        print(ans_no_cot)

        print("\n[WITH CoT]")
        ans_cot = ask_with_cot(q)
        print(ans_cot)

        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()