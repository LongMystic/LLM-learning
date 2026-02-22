# 05 – Lý thuyết và khái niệm: Fine-tuning & PEFT

Tài liệu này tóm tắt các khái niệm cần nắm trong **module 05 (Fine-tuning PEFT)**, áp dụng cho việc fine-tune LLM local/self-host.

---

## 1. Fine-tuning là gì?

- **Fine-tuning (tinh chỉnh)**: tiếp tục train một mô hình đã được pre-train (trên ngôn ngữ/task chung) bằng dữ liệu mới, thường cho **một task hoặc domain cụ thể** (ví dụ: trả lời theo phong cách công ty, dịch thuật chuyên ngành, chat theo tone riêng).
- **Mục đích**: mô hình vừa giữ kiến thức nền (từ pre-train), vừa thích nghi với dữ liệu/task mới mà không cần train từ đầu (tốn rất nhiều tài nguyên).

---

## 2. Full fine-tuning vs Parameter-Efficient Fine-Tuning (PEFT)

- **Full fine-tuning**: cập nhật **toàn bộ** tham số của mô hình. Chất lượng có thể tốt nhưng:
  - Cần nhiều VRAM và thời gian.
  - Dễ “quên” kiến thức nền (catastrophic forgetting) nếu dữ liệu mới ít hoặc lệch.
- **PEFT**: chỉ cập nhật **một phần nhỏ** tham số (adapter, low-rank matrices, …), phần còn lại giữ nguyên (frozen).
  - Ít VRAM, train nhanh hơn.
  - Dễ quản lý nhiều “phiên bản” (mỗi task một adapter nhỏ).
  - Thường dùng trong production và research hiện đại.

---

## 3. LoRA (Low-Rank Adaptation)

- **Ý tưởng**: thay vì sửa trực tiếp ma trận trọng số \( W \) (kích thước lớn), ta học **hai ma trận hạng thấp** \( A \) (d×r) và \( B \) (r×k) sao cho cập nhật có dạng \( \Delta W = B \cdot A \), với **r (rank)** nhỏ (8, 16, 32, …).
- **Lợi ích**: số tham số cần train = 2×d×r (hoặc tương đương) << d×k, nên tiết kiệm bộ nhớ và thời gian.
- **Các siêu tham số thường gặp**:
  - **r (rank)**: hạng của ma trận low-rank; tăng r → linh hoạt hơn nhưng nhiều tham số hơn.
  - **alpha (scaling)**: thường dùng tỉ lệ `alpha/r` để scale cập nhật (LoRA scaling).
  - **target_modules**: các layer nào của mô hình được gắn LoRA (VD: chỉ `q_proj`, `v_proj` trong attention).

---

## 4. QLoRA (Quantized LoRA)

- **Quantization**: nén trọng số từ 16-bit (hoặc 32-bit) xuống 8-bit hoặc 4-bit để **giảm VRAM** khi load mô hình.
- **QLoRA**: kết hợp **quantization** (base model ở 4-bit/8-bit) với **LoRA** (chỉ train adapter ở full precision hoặc higher precision).
  - Base model gần như không đổi, chỉ adapter được cập nhật.
  - Cho phép fine-tune model lớn (7B, 13B) trên GPU ít VRAM (6–12GB).
- **Công cụ thường dùng**: `bitsandbytes` (QuantizationConfig / BitsAndBytesConfig trong `transformers`).

---

## 5. Các khái niệm bổ sung

- **Adapter**: module nhỏ gắn vào mô hình (VD: LoRA là một dạng adapter). Sau khi train, chỉ cần lưu adapter (vài MB đến vài trăm MB) thay vì toàn bộ model.
- **Merge adapter**: cộng trọng số LoRA vào base model để được một mô hình “đầy đủ” (dùng khi muốn export 1 file, không cần tải base + adapter riêng).
- **Instruction tuning / SFT (Supervised Fine-Tuning)**: train trên cặp (instruction, response) để mô hình học làm theo hướng dẫn hoặc format hội thoại.
- **Catastrophic forgetting**: khi fine-tune, mô hình có thể “quên” hành vi cũ. PEFT và dữ liệu cân bằng giúp giảm hiện tượng này.

---

## 6. Áp dụng trong module 05

- **05.1**: Chuẩn bị dữ liệu instruction/response và pipeline tokenize.
- **05.2**: Dùng **LoRA** (full precision base) để làm quen với PEFT và training loop.
- **05.3**: Dùng **QLoRA** (base 4-bit) để giảm VRAM và thử trên máy ít bộ nhớ.
- **05.4**: Merge adapter (và nếu cần export) để tích hợp với pipeline inference (VD: Ollama, `transformers`).
- **05.5**: Đánh giá và ghi chép cấu hình, loss, mẫu generation — từ đó hiểu rõ ảnh hưởng của rank, alpha, và lượng dữ liệu.

Sau khi nắm lý thuyết và thực hành 05.1–05.5, bạn có thể mở rộng sang **preference optimization** (DPO, ORPO) hoặc **serving** (module 06) để đưa model đã fine-tune vào sử dụng thực tế.
