"""
07.3 – Chấm điểm đơn giản log run_once_* bằng rule-based.

- Input: logs/run_once_XXX.jsonl
- Output: in bảng ID → score, và trung bình.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

MODULE_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = MODULE_ROOT / "logs"

RUN_FILE = "run_once_llama3_2.jsonl"  # đổi cho đúng tên file muốn chấm


def load_run(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def simple_score(rec: Dict[str, Any]) -> float:
    """Rule-based rất đơn giản, chỉ để demo."""
    prompt = (rec.get("prompt") or "").lower()
    ans = (rec.get("answer") or "").lower()
    if not ans:
        return 0.0

    # ví dụ: nếu prompt hỏi về 'agentic' mà answer không chứa 'agent' thì phạt
    if "agentic" in prompt and "agent" not in ans:
        return 0.5

    # nếu prompt hỏi 'mcp' mà answer không chứa 'mcp'
    if "mcp" in prompt and "mcp" not in ans:
        return 0.5

    # còn lại cho 1.0 (đủ keyword cơ bản)
    return 1.0


def main():
    path = LOG_DIR / RUN_FILE
    recs = load_run(path)

    total = 0.0
    n = 0
    print(f"Scoring {len(recs)} records from {path}\n")
    print(f"{'ID':<5} {'score':>6}")
    print("-" * 14)

    for rec in recs:
        sid = rec.get("id", "?")
        s = simple_score(rec)
        print(f"{sid:<5} {s:6.2f}")
        total += s
        n += 1

    if n:
        print("\nAverage score:", round(total / n, 3))


if __name__ == "__main__":
    main()