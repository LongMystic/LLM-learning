"""
08.3 – Planner → Executor đơn giản (không dùng tool, chỉ LLM).

Flow:
1) Planner: nhận task lớn, trả về JSON plan nhiều bước.
2) Executor: chạy từng bước bằng LLM, tổng hợp kết quả cuối.
"""

from pathlib import Path
import sys
from typing import List, Dict, Any
import json

ROOT = Path(__file__).resolve().parent.parent.parent  # .../sota_llm_lab
GATEWAY_SRC = ROOT / "06_serving_llmops" / "src"
sys.path.append(str(GATEWAY_SRC))

from llm_gateway_client import chat_via_gateway  # type: ignore


TASKS = [
    {
        "id": "blog1",
        "goal": "Viết outline + đoạn mở bài cho một blog về Agentic AI trong project này.",
    },
    {
        "id": "report1",
        "goal": "Tóm tắt pipeline tổng thể 00–07 và viết email báo cáo ngắn cho sếp kỹ thuật.",
    },
]


def planner(goal: str, model: str = "llama3.2") -> List[Dict[str, Any]]:
    prompt = f"""
Bạn là planner.

Nhiệm vụ: chia nhỏ mục tiêu sau thành 3–5 bước rõ ràng, tuần tự.

Mục tiêu:
\"\"\"{goal}\"\"\"

Hãy trả về DUY NHẤT JSON dạng:
{{
  "steps": [
    {{"id": "s1", "desc": "..." }},
    ...
  ]
}}
"""
    msgs = [
        {"role": "system", "content": "Bạn là planner, chỉ trả về JSON hợp lệ, không thêm text khác."},
        {"role": "user", "content": prompt},
    ]
    text = chat_via_gateway(msgs, model=model, temperature=0.3, max_tokens=512).strip()
    try:
        data = json.loads(text)
        steps = data.get("steps", [])
        return steps if isinstance(steps, list) else []
    except Exception:
        # fallback: 1 bước duy nhất
        return [{"id": "s1", "desc": goal}]


def executor(goal: str, steps: List[Dict[str, Any]], model: str = "llama3.2") -> str:
    """Thực thi tuần tự từng bước bằng LLM, gom kết quả lại."""
    notes: List[str] = []
    for step in steps:
        sid = step.get("id", "?")
        desc = step.get("desc", "")
        print(f"- Executing {sid}: {desc}")

        prompt = f"""
Mục tiêu tổng:
\"\"\"{goal}\"\"\"

Bước hiện tại ({sid}):
\"\"\"{desc}\"\"\"

Dùng kiến thức sẵn có của bạn để hoàn thành bước này.
Nếu cần, bạn có thể nhắc lại ngắn gọn kết quả từ các bước trước:
\"\"\"{chr(10).join(notes[-2:])}\"\"\"
"""
        msgs = [
            {"role": "system", "content": "Bạn là executor, thực hiện từng bước nhỏ rồi ghi lại kết quả."},
            {"role": "user", "content": prompt},
        ]
        out = chat_via_gateway(msgs, model=model, temperature=0.7, max_tokens=512).strip()
        notes.append(f"[{sid}] {out}")

    # tổng hợp kết quả cuối cùng
    summary_prompt = f"""
Mục tiêu tổng:
\"\"\"{goal}\"\"\"

Ghi chú từ các bước:
\"\"\"{chr(10).join(notes)}\"\"\"

Hãy viết KẾT QUẢ CUỐI CÙNG ngắn gọn, rõ ràng cho người dùng (không liệt kê từng bước nữa).
"""
    msgs = [
        {"role": "system", "content": "Bạn tổng hợp kết quả cuối, trình bày sạch sẽ."},
        {"role": "user", "content": summary_prompt},
    ]
    final = chat_via_gateway(msgs, model=model, temperature=0.7, max_tokens=512).strip()
    return final


def main():
    for task in TASKS:
        tid = task["id"]
        goal = task["goal"]

        print("\n" + "=" * 80)
        print(f"Task ID: {tid}")
        print("Goal:", goal)

        steps = planner(goal)
        print("\n[PLAN]")
        print(json.dumps(steps, ensure_ascii=False, indent=2))

        print("\n[EXECUTION]")
        final = executor(goal, steps)
        print("\n[FINAL RESULT]")
        print(final)
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()