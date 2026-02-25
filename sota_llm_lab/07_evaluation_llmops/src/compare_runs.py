"""
07.2 – So sánh 2 lần chạy eval (2 model / 2 config) trên cùng bộ prompt.

- Input: 2 file JSONL log từ run_eval_once.py
- Output: bảng tóm tắt trên console (latency, độ dài câu trả lời, preview).
"""

import json
from pathlib import Path
from typing import Dict, Any, List

MODULE_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = MODULE_ROOT / "logs"

RUN_A = "run_once_llama3_2.jsonl"
RUN_B = "run_once_qwen3_4b.jsonl"  # đổi cho đúng tên file bạn có


def load_run(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def main():
    a_path = LOG_DIR / RUN_A
    b_path = LOG_DIR / RUN_B

    run_a = load_run(a_path)
    run_b = load_run(b_path)

    ids = sorted(set(run_a.keys()) & set(run_b.keys()))
    print(f"Found {len(ids)} common prompt IDs.\n")

    header = f"{'ID':<5} {'modelA_lat':>10} {'modelB_lat':>10} {'lenA':>6} {'lenB':>6}"
    print(header)
    print("-" * len(header))

    for pid in ids:
        ra = run_a[pid]
        rb = run_b[pid]

        ans_a = (ra.get("answer") or "").strip()
        ans_b = (rb.get("answer") or "").strip()

        lat_a = ra.get("latency_sec", 0.0)
        lat_b = rb.get("latency_sec", 0.0)

        print(
            f"{pid:<5} {lat_a:10.2f} {lat_b:10.2f} {len(ans_a):6d} {len(ans_b):6d}"
        )

    print("\n=== Preview chi tiết từng prompt (tùy chọn) ===")
    for pid in ids:
        ra = run_a[pid]
        rb = run_b[pid]
        print("\n" + "=" * 80)
        print(f"ID: {pid}")
        print("Prompt:", ra.get("prompt", ""))
        print("\n[Model A]")
        print((ra.get("answer") or "")[:400])
        print("\n[Model B]")
        print((rb.get("answer") or "")[:400])


if __name__ == "__main__":
    main()