## 06_serving_llmops – Serving LLM & LLMOps với Ollama

Mục tiêu module 06:

- **Chuẩn hoá runtime Ollama** làm backend LLM chính (thay vì tự load model lớn bằng `transformers` trên CPU).
- Xây dựng một lớp **gateway / service** (HTTP API) để các module khác (MCP, Agents, RAG, Fine-tuning, v.v.) gọi LLM một cách thống nhất.
- Đặt nền cho các bước LLMOps sau này (logging, monitoring, tracing, multi-model, A/B test…).

### Các bước dự kiến

1. **06.1 – Ollama client (Python)**  
   - Viết một client Python nhẹ để gọi Ollama qua HTTP API.  
   - Chuẩn hoá hàm kiểu `chat_ollama(messages, model=..., stream=False)`.

2. **06.2 – LLM Gateway (FastAPI)**  
   - Xây dựng một service FastAPI đơn giản:
     - Endpoint `/chat` nhận `messages` (format giống OpenAI) → gọi Ollama → trả response.
     - (Tuỳ chọn) endpoint `/chat/stream` để stream token từ Ollama.
   - Service này đóng vai trò **LLM gateway** cho toàn bộ project.

3. **06.3 – Tích hợp với các module trước**  
   - Thay các chỗ gọi model rời rạc bằng LLM gateway mới.
   - Đảm bảo MCP, Agents, RAG, Fine-tuning có thể cấu hình để dùng Ollama làm runtime chung.

4. **06.4 – Gợi ý LLMOps (tuỳ thời gian/tài nguyên)**  
   - Logging request/response LLM.
   - Ghi lại prompt/response để phân tích, gỡ lỗi.
   - Phác thảo hướng A/B test giữa nhiều model Ollama (ví dụ `llama3` vs `qwen2.5`).

### Ghi chú thực hành

- **Quan trọng**: Trong module này, các file `.py` (client, FastAPI, tích hợp) sẽ được mô tả bằng **code block trong tài liệu / message**, và bạn tự tạo file để luyện tay.
- Yêu cầu: đã cài **Ollama** và có ít nhất một model (ví dụ `llama3`) pull sẵn để test.

