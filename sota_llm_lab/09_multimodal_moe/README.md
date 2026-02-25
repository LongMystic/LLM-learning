## 09_multimodal_moe – Multimodal & Mixture-of-Experts (MoE)

Mục tiêu:
- Có **góc nhìn tổng quan** về:
  - Multimodal LLM (text + image/âm thanh).
  - Mixture-of-Experts (MoE) – nhiều “expert” cùng tồn tại, router chọn ai xử lý.
- Thực hành 1–2 ví dụ **pseudo‑MoE** dựa trên các model text sẵn có qua Ollama Gateway.

> Lưu ý: do chạy local/CPU và phụ thuộc vào model bạn đã pull trong Ollama, phần multimodal sẽ ở mức mô tả + stub code. Phần MoE sẽ được demo rõ hơn (router nhiều model text).

### Các bước dự kiến

1. **09.1 – Pseudo‑MoE router đơn giản (rule-based)**  
   - Viết một script router:
     - Query về **code / kỹ thuật** → gửi tới model A (ví dụ `llama3.2`).
     - Query về **viết lách / email** → gửi tới model B (ví dụ `qwen2.5` hoặc model khác bạn chọn).
   - Dùng `llm_gateway_client.chat_via_gateway` với tham số `model` khác nhau.

2. **09.2 – Router dùng LLM (LLM-as-router)**  
   - Thay vì rule-based đơn giản, dùng chính LLM để phân loại intent:
     - Bước 1: LLM phân loại câu hỏi vào nhóm: `code`, `writing`, `rag`, ...
     - Bước 2: chọn model tương ứng và gọi lại qua gateway.
   - Đây là “pseudo‑MoE”: router + nhiều expert model backend.

3. **09.3 – Multimodal (overview + stub)**  
   - Giải thích cách dùng LLM multimodal:
     - Text + image input (caption, QA trên ảnh).
     - Trong thực tế có thể dùng:
       - Model vision trong Ollama (nếu bạn đã pull, ví dụ `llava`, `llama3.2-vision`, ...).
       - Hoặc API ngoài (OpenAI GPT‑4o, v.v.).
   - Viết stub code mô tả input/output và để TODO kết nối thực tế tuỳ môi trường của bạn.

4. **09.4 – Liên hệ với Agentic AI (module 03)**  
   - Multimodal: thêm “channel” mới cho agent (đọc ảnh, sơ đồ, screenshot, ...).
   - MoE router: là bước tự nhiên tiếp theo của orchestrator/agent – thay vì 1 model, orchestrator có thể chọn **expert model** theo từng loại task.

> Code `.py` cho module 09 sẽ tiếp tục được mô tả bằng code block, bạn tự tạo file để luyện tay. Nhấn mạnh mục tiêu là **overview + cảm nhận pattern**, không phải build full system multimodal/MoE hoàn chỉnh.

