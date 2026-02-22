# 02 – Kiến trúc & Lược đồ

## Lược đồ tổng thể (MCP server + Ollama planner)

```
┌─────────────┐     user query      ┌─────────────────────┐
│   User      │ ──────────────────► │  Ollama Tool        │
│   (CLI)     │                     │  Planner (Python)   │
└─────────────┘                     └──────────┬──────────┘
       ▲                                       │
       │ answer + tool result                  │ 1. prompt + tools description
       │                                       ▼
       │                             ┌─────────────────────┐
       │                             │  Ollama (LLM)       │
       │                             │  localhost:11434    │
       │                             └──────────┬──────────┘
       │                                        │ 2. JSON: { "tool", "args" }
       │                                        ▼
       │                             ┌─────────────────────┐
       │                             │  MCP-like Server    │
       │                             │  (FastAPI)          │
       │                             │  localhost:8001     |
       │                             └──────────┬──────────┘
       │                                        │ 3. call_tool(name, args)
       │                                        ▼
       │                             ┌─────────────────────┐
       │                             │  tools.py           │
       │                             │  (echo, read_file,  │
       │                             │   search_files,     │
       │                             │   call_http, ...)   │
       │                             └──────────┬──────────┘
       │                                        │ 4. result (string / list)
       └────────────────────────────────────────┘
```

- **Ollama Tool Planner**: script Python tương tác với user, gửi câu hỏi + mô tả tools cho Ollama, nhận JSON `{tool, args}`, gọi MCP server qua HTTP, in kết quả.
- **MCP-like Server**: FastAPI expose endpoint `POST /call_tool`; body `{ "tool": "...", "args": {...} }` → gọi `call_tool()` trong `tools.py` → trả `{ "ok", "result" }`.
- **tools.py**: chứa các hàm thuần (echo, read_file, search_files, call_http, grep_code, read_log, query_sqlite) và dict `TOOLS` để dispatch.

## Transport (stdio / HTTP / JSON-RPC)

- **Chuẩn MCP**: thường dùng **stdio** — server chạy như subprocess, client gửi/nhận dòng JSON-RPC qua stdin/stdout. Phù hợp tích hợp IDE (Cursor, VS Code extension).
- **Module 01 (MCP-like)**: dùng **HTTP** — server là web API (FastAPI), client dùng `requests.post(url, json=...)`. Đơn giản, dễ debug, dễ dùng từ script hoặc agent Python.
- **JSON-RPC**: format request/response có `jsonrpc`, `id`, `method`, `params`. MCP chuẩn dùng JSON-RPC 2.0; implementation HTTP của ta đơn giản hóa (chỉ một “method” là call_tool).

Sau này nếu muốn tương thích Cursor/CLI MCP chuẩn, có thể thêm một server stdio đọc dòng, parse JSON-RPC và gọi cùng `tools.call_tool()`.

## Cấu trúc thư mục module 01

```
_01_mcp/
├── README.md                 # Mục tiêu, bài tập 1–4, gợi ý tổ chức
├── docs/
│   ├── 01_concepts.md        # Khái niệm MCP, so sánh, kiến trúc, luồng dữ liệu
│   ├── 02_architecture.md    # Lược đồ, transport, cấu trúc thư mục
│   ├── 03_run_guide.md       # Cách chạy server, client, planner
│   ├── 04_links.md           # Tài liệu tham khảo
│   └── 05_adding_tools.md    # Hướng dẫn thêm tool (mục 3, 4)
├── servers/
│   ├── tools.py              # Định nghĩa các tool + TOOLS dict + call_tool()
│   ├── mcp_like_server.py    # FastAPI app, endpoint POST /call_tool
│   └── ollama_tool_planner.py # Interactive loop: user → Ollama → MCP server → user
└── clients/
    └── test_client.py        # Script test gọi /call_tool trực tiếp (không qua LLM)
```

- **servers/**: toàn bộ logic MCP (tools + server HTTP + planner dùng Ollama).
- **clients/**: script test hoặc client mẫu gọi server.
- **docs/**: ghi chú học tập và hướng dẫn chạy / thêm tool.
