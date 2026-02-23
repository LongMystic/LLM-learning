"""
05.5 – Đánh giá nhanh base vs LoRA vs merged LoRA.

Chạy trên CPU, dùng vài prompt cố định và so sánh đầu ra.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from prepare_data import DEFAULT_MODEL_NAME

BASE_NAME = DEFAULT_MODEL_NAME
ROOT = Path(__file__).resolve().parent.parent
LORA_ADAPTER_DIR = ROOT / "experiments" / "lora_run" / "final_adapter"
MERGED_MODEL_DIR = ROOT / "experiments" / "merged_lora_model"

PROMPTS = [
    "Ollama là gì và liên quan đến MCP như thế nào?",
    "Orchestrator trong Agentic AI là gì?",
    "Viết một câu trả lời lịch sự cho khách hàng đang phàn nàn về việc giao hàng chậm.",
    "Tóm tắt ngắn gọn mục tiêu của module 05 trong project này.",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # Cắt phần prompt đi cho dễ đọc
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()


def load_base():
    print("Load BASE model...")
    tok = AutoTokenizer.from_pretrained(BASE_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def load_lora():
    if not LORA_ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy LoRA adapter tại {LORA_ADAPTER_DIR}. Chạy 05.2 trước.")
    print("Load BASE + LoRA adapter...")
    tok = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_ADAPTER_DIR)
    model.eval()
    return model, tok


def load_merged():
    if not MERGED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy merged model tại {MERGED_MODEL_DIR}. Chạy merge_lora_adapter.py (05.4) trước.")
    print("Load MERGED LoRA model...")
    tok = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def main():
    base_model, base_tok = load_base()
    lora_model, lora_tok = load_lora()
    merged_model, merged_tok = load_merged()

    for i, prompt in enumerate(PROMPTS, start=1):
        print("=" * 80)
        print(f"Prompt {i}: {prompt}\n")

        base_out = generate(base_model, base_tok, prompt)
        print("[BASE]")
        print(base_out, "\n")

        lora_out = generate(lora_model, lora_tok, prompt)
        print("[LoRA adapter 05.2]")
        print(lora_out, "\n")

        merged_out = generate(merged_model, merged_tok, prompt)
        print("[Merged LoRA 05.4]")
        print(merged_out, "\n")

    print("=" * 80)
    print("Hoàn thành 05.5 – hãy xem sự khác biệt giữa BASE, LoRA và MERGED trên các prompt trên.")


if __name__ == "__main__":
    main()

