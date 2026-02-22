# Giải thích: prepare_data.py – Luồng và vai trò

## 1. Script này làm gì?

`prepare_data.py` thực hiện **bước chuẩn bị dữ liệu** cho fine-tuning (LoRA/QLoRA): đọc file instruction/response, ghép thành text theo template, rồi **tokenize** bằng tokenizer của model gốc. Kết quả là một **Dataset** đã có `input_ids`, `attention_mask`, `labels` — sẵn sàng đưa vào training loop ở bước 05.2.

---

## 2. Input

| Input | Mô tả |
|-------|--------|
| **File JSONL** | Đường dẫn mặc định: `05_finetuning_peft/data/sample_instructions.jsonl`. Mỗi dòng là một JSON với hai key: `instruction` (câu hỏi / yêu cầu) và `response` (câu trả lời mẫu). |
| **Tên model** | Dùng để tải **tokenizer** (không tải model). Mặc định: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Tokenizer quyết định từ vựng và cách chuyển text → token id. |
| **max_length** | Độ dài tối đa mỗi mẫu sau khi tokenize (số token). Mặc định 512. Câu dài hơn sẽ bị cắt (truncation), ngắn hơn sẽ được pad. |

Ví dụ một dòng trong file JSONL:

```json
{"instruction": "MCP là gì?", "response": "MCP (Model Context Protocol) là giao thức..."}
```

---

## 3. Xử lý bên trong

1. **Đọc JSONL** → list các dict `{instruction, response}`.
2. **Ghép mỗi cặp thành một đoạn text** theo template:
   ```
   ### Instruction:
   {instruction}

   ### Response:
   {response}
   ```
3. **Tokenize** toàn bộ đoạn text:
   - Dùng tokenizer của base model (TinyLlama hoặc model bạn chỉnh).
   - `truncation=True`, `max_length=512`, `padding="max_length"` → mọi mẫu đều có cùng độ dài (512 token).
   - Tạo thêm cột `labels`: bản sao của `input_ids` (dùng làm target khi train language model).
4. **Định dạng** dataset sang PyTorch (`set_format("torch")`) để dùng trực tiếp với `DataLoader` / `Trainer`.

---

## 4. Output

- **Kiểu**: `datasets.Dataset` (Hugging Face).
- **Các cột**:
  - `input_ids`: list/tensor token id, length = max_length (512).
  - `attention_mask`: mask (1 = token thật, 0 = padding).
  - `labels`: giống input_ids (hoặc sau này có thể mask phần instruction để chỉ train trên phần response).

Output này là **input chính** cho bước 05.2 (train LoRA): script train sẽ gọi `prepare_dataset()` để lấy dataset, rồi đưa vào `Trainer` hoặc vòng lặp train.

---

## 5. Phục vụ các bước tiếp theo như thế nào?

| Bước | Cách dùng output của prepare_data.py |
|------|--------------------------------------|
| **05.2 LoRA** | Gọi `prepare_dataset()` → nhận Dataset đã tokenize. Load base model + gắn LoRA, tạo `DataLoader` từ dataset, train với loss trên `labels`. |
| **05.3 QLoRA** | Giống 05.2, nhưng base model load ở dạng 4-bit; dataset vẫn dùng chung từ `prepare_dataset()`. |
| **05.4 Merge / export** | Không dùng trực tiếp prepare_data; dùng model đã train từ 05.2/05.3. |
| **05.5 Đánh giá** | Có thể gọi lại `prepare_dataset()` để lấy thêm validation set hoặc vài mẫu cố định để so sánh generation trước/sau fine-tune. |

Tóm lại: **prepare_data.py** chuẩn hóa dữ liệu từ “instruction + response” sang dạng token id + mask + labels, thống nhất với tokenizer của base model, để mọi bước train và đánh giá sau này dùng chung một định dạng.
