## 02_Agents – Xây dựng Agents (Local / Self-host)

Mục tiêu: hiểu khái niệm **AI Agent**, các thành phần chính và **tự build** agent chạy local có thể chọn tool, lập kế hoạch đơn giản và tương tác nhiều bước.

### Kiến thức cần nắm

- **Định nghĩa Agent**
  - Agent = LLM + Observation + Action + Memory (tối thiểu).
  - Vòng lặp: nhận mục tiêu → suy nghĩ (reasoning) → chọn hành động (tool) → quan sát → lặp lại.
- **Cấu trúc 1 agent đơn giản**
  - Planner (có thể là prompt nội bộ trong LLM).
  - Tooling layer (MCP / function calling / custom tools).
  - State & memory (context, history, intermediate results).
- **Các pattern phổ biến**
  - ReAct (Reason + Act).
  - Toolformer / Function calling-based agents.
  - Single-agent vs multi-agent.

### Thực hành (gợi ý bài tập)

1. **Agent CLI 1-tool**
   - Agent có 1 tool duy nhất (VD: search file / tính toán).
   - Vòng lặp:
     - User nhập mục tiêu.
     - Agent suy nghĩ (hiển thị “thoughts” trong console).
     - Agent gọi tool 1 lần, trả kết quả cuối.

2. **Agent nhiều tool (local)**
   - Thêm 2–3 tool (VD: đọc file, gọi HTTP, tính toán Python).
   - Cho phép agent tự chọn tool dựa trên prompt nội bộ.
   - Log lại chuỗi hành động (action trace) để debug.

3. **Agent có memory ngắn hạn**
   - Lưu context hội thoại + intermediate results.
   - Cho agent có thể tham chiếu lại kết quả bước trước, không cần hỏi lại user.

4. **Mini-project: Task Assistant agent**
   - Agent giúp bạn làm 1 task cụ thể (ví dụ):
     - Đọc tài liệu trong 1 thư mục, tóm tắt, trả lời câu hỏi.
     - Phân tích 1 file CSV, tạo báo cáo text.
   - Agent phải:
     - Tự lên vài bước hành động.
     - Tự gọi các tool cần thiết (qua MCP / function calling).

### Gợi ý tổ chức thư mục

- `docs/`: note về khái niệm agent, diagrams, prompt templates.
- `simple_agents/`: ví dụ agent đơn.
  - `cli_agent_no_llm.py`: agent CLI chọn tool bằng menu (02.1).
  - `ollama_react_agent.py`: ReAct agent dùng Ollama, nhiều bước (02.2).
  - `ollama_react_agent_with_memory.py`: ReAct + memory ngắn hạn + action trace log (02.3). Trace ghi tại `simple_agents/logs/`.
- `multi_agent/`: agent nhiều vai (planner, executor, reviewer…).
- `tools/`: code các tool có thể tái sử dụng (I/O, HTTP, DB, MCP wrappers).

