# 04 – Tài liệu tham khảo

## Links chính thức MCP

- **Spec & GitHub**: [Model Context Protocol (MCP) – Anthropic](https://modelcontextprotocol.io/)  
  Trang chủ protocol: giới thiệu, mục đích, kiến trúc tổng quan.

- **GitHub – modelcontextprotocol**:  
  https://github.com/modelcontextprotocol  
  Repo chính: spec chi tiết, SDK (Python, TypeScript), danh sách server/client mẫu.

- **MCP Python SDK**:  
  https://github.com/modelcontextprotocol/python-sdk  
  Dùng khi muốn viết MCP server/client chuẩn (stdio, JSON-RPC) bằng Python.

- **MCP Servers (danh sách)**:  
  Trong repo MCP hoặc trang docs thường có link tới các server mẫu (filesystem, GitHub, database, …) để tham khảo cách định nghĩa tools và resources.

## Tài liệu bổ sung

- **Ollama API**: https://github.com/ollama/ollama/blob/main/docs/api.md  
  Tài liệu REST API của Ollama (`/api/generate`, `/api/chat`, …) dùng cho planner.

- **FastAPI**: https://fastapi.tiangolo.com/  
  Tài liệu framework dùng cho MCP-like HTTP server.

- **Cursor / IDE tích hợp MCP**:  
  Cursor và một số IDE hỗ trợ MCP client; cấu hình thêm MCP server (stdio) sẽ cho phép model trong IDE gọi tools của bạn. Xem docs Cursor hoặc VS Code extension MCP nếu muốn gắn server module 01 vào editor.

- **RAG, Agents**: Sau khi nắm MCP, có thể đọc thêm tài liệu RAG (Retrieval-Augmented Generation) và Agent (ReAct, tool use) để thấy cách MCP được dùng trong pipeline agent đa bước (module 02, 03).
