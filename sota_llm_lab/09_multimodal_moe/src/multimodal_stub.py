"""
09.3 – Multimodal stub: minh hoạ cách gửi text + image tới model vision (Ollama hoặc API khác).

Lưu ý:
- File này chỉ là khung. Tuỳ model vision thực tế (llava, llama3-vision, GPT-4o, ...),
  bạn cần chỉnh lại endpoint, payload và cách đọc response cho đúng docs.
"""

import base64
from pathlib import Path
from typing import Dict, Any

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
# Ví dụ API native: /api/chat hoặc /api/generate, tuỳ model hỗ trợ
VISION_URL = f"{OLLAMA_BASE_URL}/api/chat"
VISION_MODEL = "llava"  # ĐỔI theo model multimodal bạn có trong Ollama


def encode_image_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def ask_vision_model(image_path: str, question: str) -> Dict[str, Any]:
    """
    Stub: minh hoạ payload. Hãy đọc docs model vision cụ thể để chỉnh lại.
    """
    img_b64 = encode_image_to_base64(Path(image_path))

    # Payload này CHỈ là ví dụ, nhiều model dùng schema khác (vd: "images": [...])
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": question,
                # Ví dụ chỗ để đính kèm ảnh, tuỳ API:
                # "images": [img_b64],
            }
        ],
        "stream": False,
    }

    resp = requests.post(VISION_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    print("=== Multimodal Stub ===")
    print(
        "Đây chỉ là khung. Khi bạn có model vision cụ thể (Ollama hoặc API ngoài), "
        "hãy đọc docs của nó và chỉnh payload trong ask_vision_model()."
    )
    # Ví dụ sau khi đã chỉnh payload cho đúng:
    # result = ask_vision_model(\"sample.jpg\", \"Mô tả ngắn gọn nội dung bức ảnh này.\")
    # print(result)


if __name__ == \"__main__\":
    main()

