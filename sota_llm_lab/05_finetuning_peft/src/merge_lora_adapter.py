"""
05.4 – Merge LoRA adapter (05.2) vào base model TinyLlama và export model đã merge.

- Input:
  - Base model: DEFAULT_MODEL_NAME (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  - Adapter: experiments/lora_run/final_adapter
- Output:
  - Model đã merge: experiments/merged_lora_model
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from prepare_data import DEFAULT_MODEL_NAME

ADAPTER_DIR = Path(__file__).resolve().parent.parent / "experiments" / "lora_run" / "final_adapter"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "merged_lora_model"


def main():
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy adapter tại {ADAPTER_DIR}. Hãy chạy train_lora.py (05.2) trước.")

    print("Load base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Load LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merge LoRA vào base model...")
    merged_model = model.merge_and_unload()

    print(f"Lưu model đã merge vào {OUTPUT_DIR} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)

    print("Lưu tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Model đã merge nằm ở: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

