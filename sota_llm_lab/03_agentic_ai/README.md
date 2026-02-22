## 03_Agentic_AI – Thiết kế Hệ thống Agentic AI

Mục tiêu: từ agents rời rạc (module 02) tiến lên **hệ thống Agentic AI hoàn chỉnh**: nhiều agent, workflow, memory, orchestration và giám sát.

### Kiến thức cần nắm

- **Agentic AI là gì?**
  - Hệ thống gồm nhiều agent phối hợp (hoặc 1 agent phức tạp) để giải quyết bài toán mở.
  - Tập trung vào: decomposition (chia nhỏ nhiệm vụ), planning, execution, feedback loop.
- **Thành phần của 1 hệ agentic**
  - Coordinator / Orchestrator (có thể là 1 agent khác).
  - Các worker agents chuyên trách (researcher, coder, reviewer, data-analyst…).
  - Memory & knowledge base (RAG, DB, file system).
  - Tool layer (MCP, HTTP, DB, code execution, …).
- **Patterns phổ biến**
  - Planner–Executor–Critic.
  - Graph-based workflows (DAG) giữa các agent.
  - Self-reflection / self-correction.

### Thực hành (gợi ý bài tập)

1. **Mini multi-agent workflow**
   - 2 agents:
     - `research_agent`: tìm thông tin, tóm tắt.
     - `writer_agent`: viết lại nội dung thành báo cáo cho người dùng.
   - Orchestrator:
     - Nhận yêu cầu user.
     - Gửi subtask cho `research_agent`, sau đó pass kết quả cho `writer_agent`.

2. **Agentic system với RAG + tools**
   - Kết hợp RAG local của bạn (module 04 sau này) + các tool MCP.
   - Ví dụ:
     - Agent đọc docs local, query RAG, rồi quyết định có cần gọi HTTP API bổ sung không.

3. **Feedback & self-reflection**
   - Thêm 1 agent hoặc 1 bước “critic”:
     - Review kết quả cuối, so với yêu cầu ban đầu, đề xuất chỉnh sửa.
   - Cho phép vòng lặp tối đa N lần tự sửa trước khi trả kết quả cho user.

4. **Project: Personal AI Operator (local)**
   - Hệ thống có thể:
     - Đọc/tóm tắt tài liệu từ 1 folder.
     - Tạo báo cáo / email / ghi chú.
     - Ghi log actions vào file để debug.
   - Chạy hoàn toàn local/self-host (hoặc chỉ dùng API LLM nhưng mọi orchestrations + tools là local).

### Gợi ý tổ chức thư mục

- `docs/`: kiến trúc tổng thể, sơ đồ workflow, lesson learned.
- `patterns/`: cài đặt lại các pattern (planner–executor–critic, multi-agent pipeline…).
- `projects/`: 1–2 project agentic hoàn chỉnh (ưu tiên build thật để dùng hằng ngày).

