## 01_MCP – Model Context Protocol (Local / Self-host)

Mục tiêu: hiểu và **tự xây** các MCP server/client để LLM có thể dùng công cụ, API và tài nguyên local của bạn một cách chuẩn hóa.

### Kiến thức cần nắm

- **Khái niệm MCP**:
  - MCP là gì, giải quyết vấn đề gì trong hệ sinh thái LLM.
  - So sánh MCP với việc gọi API/tool “thủ công”.
- **Kiến trúc MCP**:
  - MCP server, client, transport (thường là stdio / JSON-RPC).
  - Khái niệm tools, prompts, resources trong MCP.
- **Luồng dữ liệu**:
  - LLM → MCP client → MCP server → external tool/service → trả kết quả về LLM.

### Thực hành (gợi ý bài tập)

1. **Hello MCP server**
   - Viết 1 MCP server đơn giản (VD: Node/Python) với 1 tool:
     - `echo_tool`: nhận text và trả về text đó.
   - Kết nối với client (VD: IDE/CLI hỗ trợ MCP) và gọi thử tool.

2. **Kết nối tới tài nguyên local**
   - Viết MCP server:
     - Tool `read_file`: đọc nội dung file trong 1 thư mục cho phép.
     - Tool `search_files`: tìm file theo pattern (VD: *.py) trong thư mục code của bạn.

3. **Kết nối tới dịch vụ ngoài**
   - Viết MCP server có tool:
     - `call_http`: gọi 1 REST API public (VD: weather API, crypto price).
   - Cho phép LLM tự chọn tool để trả lời câu hỏi dựa trên API.

4. **Mini-project**
   - Xây 1 “MCP toolbox” cho chính máy của bạn:
     - Tool đọc/Grep code.
     - Tool đọc file log.
     - Tool query 1 DB local (VD: SQLite).

### Gợi ý tổ chức thư mục

- `docs/`: ghi chú concepts MCP, lược đồ kiến trúc, links tài liệu. **Khung mục chính**: xem `docs/01_concepts.md`, `02_architecture.md`, `03_run_guide.md`, `04_links.md` (tự điền nội dung).
- `servers/`: code MCP servers và tools.
- `clients/`: script/client mẫu để test (CLI, notebook, etc.).
- `examples/`: ví dụ usage, demo scripts, prompt mẫu.

**Thêm tool cho mục 3 (call_http) và mục 4 (grep, log, SQLite)**: xem hướng dẫn từng bước trong `docs/05_adding_tools.md`. Code mẫu đã có trong `servers/tools.py`; chỉ cần cập nhật mô tả tool trong `servers/ollama_tool_planner.py`.

