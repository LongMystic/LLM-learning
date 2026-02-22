# 03 – Hướng dẫn chạy

## Chuẩn bị (Ollama, Python, port)

1. **Ollama**: Cài và chạy Ollama, pull ít nhất một model (ví dụ `ollama pull llama3.2`). Đảm bảo service chạy tại `http://localhost:11434`.
2. **Python**: Dùng Python 3.10+ (khuyến nghị 3.12). Tạo venv nếu muốn tách môi trường:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```
3. **Cài dependency** (từ thư mục repo hoặc `_01_mcp/`):
   ```bash
   pip install fastapi uvicorn pydantic requests
   ```
4. **Port**: MCP server mặc định chạy trên một port (ví dụ 8001 hoặc 8081). Đảm bảo port đó không bị ứng dụng khác chiếm; nếu đổi port thì sửa `MCP_SERVER_URL` trong `ollama_tool_planner.py` cho khớp.

## Chạy MCP server (FastAPI)

1. Mở terminal, cd vào thư mục chứa code server (ví dụ `_01_mcp/servers/`).
2. Chạy:
   ```bash
   uvicorn mcp_like_server:app --reload --port 8081
   ```
   (Nếu bạn đặt tên file khác hoặc dùng port 8081, đổi cho đúng.)
3. Kiểm tra: mở trình duyệt hoặc `curl http://localhost:8001/docs` — sẽ thấy Swagger UI của FastAPI. Endpoint cần dùng là `POST /call_tool`.

## Chạy test client

1. Đảm bảo MCP server đang chạy (uvicorn như trên).
2. Trong `clients/` (hoặc từ thư mục có `test_client.py`), chỉnh `BASE_URL` trong script nếu bạn dùng port khác 8001.
3. Chạy:
   ```bash
   python test_client.py
   ```
4. Script sẽ gọi lần lượt các tool (echo, read_file, search_files,…) và in kết quả. Dùng để xác nhận server và tools hoạt động trước khi dùng planner.

## Chạy Ollama tool planner (interactive loop)

1. Đảm bảo **Ollama** đang chạy và **MCP server** đang chạy (hai terminal riêng).
2. Trong thư mục `servers/`, chạy:
   ```bash
   python ollama_tool_planner.py
   ```
3. Trong prompt, gõ câu lệnh tiếng Việt hoặc tiếng Anh (ví dụ: “Đọc file README.md”, “Tìm các file .py trong thư mục cha”). Planner sẽ gửi câu hỏi + mô tả tools cho Ollama, nhận JSON tool/args, gọi server và in kết quả.
4. Thoát: gõ `exit` hoặc `quit`.

## Ví dụ câu lệnh user có thể gõ trong loop

- **Đọc file**: “Đọc file README.md”, “Mở file servers/tools.py và cho tôi xem 50 dòng đầu”.
- **Tìm file**: “Liệt kê tất cả file .py trong thư mục hiện tại”, “Tìm file .md trong docs”.
- **Echo**: “Lặp lại câu: Học MCP rất vui.”
- **Grep (nếu đã thêm tool)**: “Tìm trong thư mục servers các dòng chứa từ `def call_tool`.”
- **Log (nếu đã thêm tool)**: “Đọc 20 dòng cuối của file app.log.”
- **HTTP (nếu đã thêm tool)**: “Gọi API thời tiết tại [URL]” (cần cấu hình URL whitelist nếu bạn đã thêm kiểm tra bảo mật).
- **SQLite (nếu đã thêm tool)**: “Chạy SELECT * FROM users LIMIT 10 trên file data/app.db.”

Tùy model và cách bạn mô tả tools trong `ollama_tool_planner.py`, có thể cần diễn đạt rõ ràng (ví dụ nêu rõ tên file hoặc đường dẫn) để Ollama chọn đúng tool và args.
