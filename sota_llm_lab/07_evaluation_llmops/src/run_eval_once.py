"""
07.1 – Chạy bộ prompt test với 1 model qua LLM Gateway, log kết quả ra JSONL.

- Input: data/prompts.jsonl
- Output: logs/run_once_MODELNAME.jsonl
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# Thêm đường dẫn tới module 06_serving_llmops/src
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent  # .../sota_llm_lab
GATEWAY_SRC = ROOT / "06_serving_llmops" / "src"
sys.path.append(str(GATEWAY_SRC))

from llm_gateway_client import chat_via_gateway  # dùng lại module 06

MODULE_ROOT = THIS_DIR.parent  # .../07_evaluation_llmops
DATA_PATH = MODULE_ROOT / "data" / "prompts.jsonl"
LOG_DIR = MODULE_ROOT / "logs"
MODEL_NAME = "qwen3:4b"


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"run_once_{MODEL_NAME.replace('.', '_').replace(':', '_')}.jsonl"

    prompts = load_prompts(DATA_PATH)
    print(f"Loaded {len(prompts)} prompts from {DATA_PATH}")

    with out_path.open("w", encoding="utf-8") as fw:
        for item in prompts:
            pid = item["id"]
            prompt = item["prompt"]
            category = item.get("category", "")

            msgs = [
                {"role": "system", "content": "Bạn là trợ lý kỹ thuật, trả lời ngắn gọn, rõ ràng."},
                {"role": "user", "content": prompt},
            ]

            t0 = time.time()
            try:
                answer = chat_via_gateway(
                    msgs,
                    model=MODEL_NAME,
                    temperature=0.7,
                    max_tokens=512,
                )
                elapsed = time.time() - t0
                ok = True
                error = None
            except Exception as e:
                answer = ""
                elapsed = time.time() - t0
                ok = False
                error = str(e)

            record = {
                "id": pid,
                "category": category,
                "model": MODEL_NAME,
                "prompt": prompt,
                "answer": answer,
                "ok": ok,
                "error": error,
                "latency_sec": elapsed,
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{pid}] ok={ok} latency={elapsed:.2f}s")

    print(f"Done. Logged to {out_path}")


if __name__ == "__main__":
    main()