# 01 – Khái niệm & Kiến thức

## Khái niệm MCP

**Model Context Protocol (MCP)** là một giao thức mở do Anthropic đề xuất, dùng để kết nối ứng dụng (IDE, chatbot, agent) với các nguồn dữ liệu và công cụ bên ngoài một cách chuẩn hóa.

- **Vấn đề MCP giải quyết**: LLM và ứng dụng cần đọc file, gọi API, query DB, chạy lệnh… Mỗi nơi tự implement một kiểu → khó tái sử dụng, khó bảo trì. MCP định nghĩa một **chuẩn chung** cho server (cung cấp tools/resources) và client (gọi từ ứng dụng/LLM).
- **Tinh thần**: MCP server “bọc” các tài nguyên (file system, API, DB) thành **tools** và **resources** có mô tả rõ ràng; client chỉ cần biết tên tool và tham số, không cần biết chi tiết triển khai.

## So sánh MCP với gọi API / tool thủ công

| | Gọi API/tool thủ công | Dùng MCP (hoặc MCP-like) |
|--|------------------------|---------------------------|
| **Định dạng** | Mỗi project tự quy ước (REST, JSON, …) | Chuẩn chung: list tools, call tool với args, trả kết quả |
| **Tái sử dụng** | Code gọi tool dính chặt với từng dịch vụ | Cùng một client có thể nói chuyện với nhiều MCP server khác nhau |
| **Mô tả tool** | Thường nằm trong code hoặc doc rời | Server cung cấp schema (tên, mô tả, tham số) → LLM/agent dễ chọn đúng tool |
| **Bảo trì** | Thay đổi API = sửa nhiều chỗ | Thay đổi ở server; client chỉ cần gọi đúng protocol |

Trong module này ta dùng **MCP-like** (HTTP + JSON) chứ chưa đủ chuẩn MCP đầy đủ (stdio/JSON-RPC), nhưng đủ để hiểu luồng và sau này nâng cấp lên MCP chuẩn.

## Kiến trúc MCP

- **MCP Server**: process cung cấp **tools** (hành động: đọc file, gọi API, …) và có thể **resources** (nội dung theo URI), **prompts** (template prompt). Server lắng nghe qua transport (stdio hoặc HTTP).
- **MCP Client**: nằm trong ứng dụng (IDE, agent, CLI). Gửi request “list tools”, “call tool X với args Y” theo đúng format protocol.
- **Transport**: kênh giao tiếp giữa client và server — thường là **stdio** (server chạy như subprocess, đọc/ghi stdin/stdout) hoặc **HTTP** (server là web API). Một số implementation dùng WebSocket.
- **LLM / Agent**: không nói trực tiếp với MCP server; nó nhận mô tả tools từ client, quyết định gọi tool nào, client chuyển tiếp lên server và trả kết quả lại cho LLM.

## Tools, Prompts, Resources

- **Tools**: hành động có thể gọi (ví dụ: `read_file`, `search_files`, `call_http`). Mỗi tool có tên, mô tả, danh sách tham số (name, type, mô tả). Client gửi `call_tool(name, args)` → server thực thi và trả kết quả.
- **Resources**: nội dung đọc được theo URI (ví dụ: `file:///path/to/file`). Dùng khi client/LLM cần “đọc” dữ liệu theo chuẩn, không nhất thiết qua tool.
- **Prompts**: template prompt có sẵn trên server (ví dụ: “summarize”, “translate”). Client có thể lấy prompt theo tên và điền biến.

Trong module 01 ta tập trung vào **tools**; resources và prompts có thể bổ sung sau.

## Luồng dữ liệu (LLM → Client → Server → Tool)

1. **User** đưa yêu cầu (ví dụ: “Đọc file README.md”).
2. **Client** (script Python hoặc agent) gửi yêu cầu đó cho **LLM** (Ollama), kèm mô tả các tool có sẵn.
3. **LLM** trả về quyết định: gọi tool nào, với tham số gì (ví dụ: `{"tool": "read_file", "args": {"path": "README.md"}}`).
4. **Client** gửi request **Call Tool** tới **MCP server** (HTTP POST với JSON trên).
5. **MCP server** gọi hàm tương ứng trong code (ví dụ `read_file(path="README.md")`), nhận kết quả.
6. **Server** trả kết quả (nội dung file) về **client**.
7. **Client** đưa kết quả lại cho **LLM** hoặc trả trực tiếp cho **user**.

Như vậy LLM không gọi OS hay API trực tiếp; mọi thao tác đều đi qua MCP server, dễ kiểm soát bảo mật và thống nhất format.
