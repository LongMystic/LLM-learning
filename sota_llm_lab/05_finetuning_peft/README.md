## 05_Finetuning_PEFT – Fine-tune LLM với PEFT (Local / Self-host)

Mục tiêu: hiểu **Parameter-Efficient Fine-Tuning (PEFT)** và tự thực hành fine-tune một mô hình ngôn ngữ (hoặc dùng base có sẵn) cho domain/task cụ thể, chạy trên máy local (có hoặc không GPU).

### Việc cần làm trong module 05

1. **05.1 – Chuẩn bị môi trường và dữ liệu**
   - Cài đặt `transformers`, `peft`, `datasets`, `accelerate` (và `bitsandbytes` nếu dùng quantization).
   - Chuẩn bị dataset: format instruction/response (VD: JSONL, hoặc HuggingFace `datasets`).
   - Script load và tiền xử lý dữ liệu (tokenize, max_length, padding).

2. **05.2 – LoRA cơ bản**
   - Load base model (VD: TinyLlama, Phi-2, hoặc model nhỏ tương thích).
   - Gắn LoRA adapter (rank r, alpha, target modules).
   - Training loop (hoặc dùng `Trainer`/`SFTTrainer`), lưu checkpoint.
   - Test inference với adapter đã train.

3. **05.3 – QLoRA (quantization + LoRA)** — script: `src/train_qlora.py`
   - Load model ở dạng 4-bit (hoặc 8-bit) với `BitsAndBytesConfig` (cần `bitsandbytes`, khuyến nghị GPU/WSL).
   - Train LoRA trên model đã quantize để giảm VRAM. Trên CPU/Windows: fallback LoRA full precision, lưu vào `experiments/qlora_run`.
   - Chi tiết: `docs/03_qlora_flow.md`. So sánh VRAM và chất lượng ở 05.5.

4. **05.4 – Merge adapter và export**
   - Merge LoRA weights vào base model (optional) để có 1 file duy nhất.
   - Xuất sang format Ollama/GGUF (optional) hoặc lưu qua HuggingFace để dùng với `pipeline`.

5. **05.5 – Đánh giá và ghi chép**
   - Đánh giá đơn giản (loss, vài mẫu generation trước/sau).
   - Ghi chép cấu hình (rank, alpha, batch size, epochs), kết quả vào `docs/` hoặc `experiments/`.

### Gợi ý tổ chức thư mục

- `docs/`: lý thuyết PEFT, khái niệm LoRA/QLoRA, ghi chép thí nghiệm (xem `docs/01_concepts_theory.md`).
- `notebooks/`: notebook thử nghiệm từng bước (load data, train LoRA, eval).
- `src/`: script train, config, utils (data load, tokenize).
- `data/`: dataset (raw hoặc processed), không commit file nặng (dùng `.gitignore`).
- `experiments/`: checkpoint, log, kết quả từng lần chạy.

### Lưu ý

- Fine-tune full model trên CPU rất chậm; ưu tiên LoRA/QLoRA và chạy trên GPU nếu có.
- Chọn base model nhỏ (7B trở xuống) để thử nhanh trên 1 GPU 8–12GB.
