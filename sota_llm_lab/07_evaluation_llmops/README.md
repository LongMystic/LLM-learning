## 07_evaluation_llmops – Đánh giá LLM & LLMOps cơ bản

Mục tiêu:
- Có cách **đo lường chất lượng** LLM (ít nhất là thủ công + bán tự động).
- Có **logging tối thiểu** cho request/response LLM để dễ debug và cải thiện.
- Đặt nền cho các bước LLMOps sau này (monitoring, A/B test đơn giản).

### Các bước dự kiến

1. **07.1 – Bộ prompt test & baseline**
   - Định nghĩa một bộ `prompts.jsonl` nhỏ (5–20 prompt) đại diện cho use case của bạn.
   - Viết script gọi 1 model (ví dụ Ollama `llama3.2` qua LLM Gateway) và lưu kết quả + thời gian.

2. **07.2 – So sánh nhiều model / version**
   - Chạy cùng bộ prompt đó với:
     - Model A: base (hoặc Ollama model 1),
     - Model B: model khác (hoặc config khác),
   - Lưu kết quả vào file/SQLite để so sánh.

3. **07.3 – Chấm điểm đơn giản**
   - Dùng rule-based (regex, keyword) hoặc LLM-as-judge đơn giản để chấm:
     - Đúng format chưa?
     - Có trả lời đủ ý chưa? (ở mức sơ bộ)
   - Tổng hợp thành bảng “prompt → model A/B → score”.

4. **07.4 – Logging & quan sát**
   - Thiết kế format log tối thiểu: thời gian, model, prompt, response, latency.
   - Gợi ý cách xem lại log để tìm bug, prompt xấu, model hành xử lạ.

> Trong module này, code `.py` sẽ được đưa trong message (code block), bạn tự tạo file để luyện tay.

