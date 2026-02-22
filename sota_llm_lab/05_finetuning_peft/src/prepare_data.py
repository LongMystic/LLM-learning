"""
05.1 – Load và tiền xử lý dữ liệu instruction/response cho fine-tune.

- Input: file JSONL (mỗi dòng {"instruction": "...", "response": "..."}), tên model (tokenizer), max_length.
- Xử lý: ghép instruction+response theo template → tokenize (truncate/pad) → thêm labels.
- Output: datasets.Dataset với input_ids, attention_mask, labels (sẵn cho Trainer/DataLoader).
- Phục vụ: 05.2 LoRA và 05.3 QLoRA gọi prepare_dataset() để lấy dataset train; 05.5 có thể dùng cho eval.
Chi tiết: xem docs/02_prepare_data_flow.md.
"""
import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_LENGTH = 512
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def build_text(ex: dict, template: str = PROMPT_TEMPLATE) -> str:
    return template.format(
        instruction=ex.get("instruction", ""), 
        response=ex.get("response", "")
    )

def prepare_dataset(
    data_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """
    Load JSONL -> tạo cột 'text' (instruction + response) -> tokenize.
    Trả về Dataset với 'input_ids', 'attention_mask', 'labels' (labels = input_ids, có thể mask phần instruction sau).
    """
    path = data_path or DATA_DIR / "sample_instructions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    rows = load_jsonl(path)
    texts = [build_text(r) for r in rows]
    raw = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        out["labels"] = out["input_ids"].copy()
        return out
    
    # num_proc=1 tránh lỗi ResourceTracker khi thoát process trên Windows
    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"], num_proc=1)
    tokenized.set_format("torch")
    print(f"Đã load {len(rows)} mẫu, tokenize với max_length={max_length}, model={model_name}")
    return tokenized

if __name__ == "__main__":
    ds = prepare_dataset()
    print(ds)
    print("Mẫu input_ids[0][:50]:", ds[0]["input_ids"][:50])
