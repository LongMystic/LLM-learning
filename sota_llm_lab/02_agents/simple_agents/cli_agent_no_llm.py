import requests
import json

MCP_SERVER_URL = 'http://localhost:8081/call_tool'

TOOLS = {
    "1": ("echo", {"text": "Nhập nội dung cần echo"}),
    "2": ("read_file", {"path": "Đường dẫn file", "max_chars": "Số ký tự tối đa (vd 2000)"}),
    "3": ("search_files", {"root": "Thư mục gốc", "pattern": "Pattern (vd *.py)", "max_results": "Số file tối đa"}),
    # "4": ("call_http", {"url": "Nhập URL cần gọi", "method": "Nhập phương thức gọi", "headers": "Nhập headers", "body": "Nhập body"}),
    "5": ("grep_code", {"path": "Đường dẫn file hoặc thư mục cần tìm", "pattern": "Pattern (vd def call_tool)", "max_lines": "Số dòng tối đa"}),
    "6": ("read_log", {"path": "Đường dẫn file", "last_n_lines": "Số dòng cuối cùng cần đọc"}),
    "7": ("query_sqlite", {"db_path": "Đường dẫn file SQLite", "query": "Câu query (vd SELECT * FROM test)"}),
}


def call_remote_tool(tool: str, args: dict):
    resp = requests.post(MCP_SERVER_URL, json={'tool': tool, 'args': args})
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(data.get("error"))
    print(f"Data: {data}")
    return data.get("result")


def ask_args(arg_schema: dict) -> dict:
    result = {}
    for key, desc in arg_schema.items():
        val = input(f"{desc}: ").strip()
        if val == "" and key in ("max_chars", "max_results", "max_lines", "last_n_lines"):
            continue
        # simple cast types
        if key in ("max_chars", "max_results", "max_lines", "last_n_lines"):
            val = int(val)
        result[key] = val
    return result


def main():
    print("== Simple MCP Agent (no LLM) ==")
    while True:
        print("\nChọn tool:")
        for k, (name, _) in TOOLS.items():
            print(f"{k}. {name}")
        print("q. Thoát")

        choice = input(">> ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            break
        if choice not in TOOLS:
            print("Lựa chọn không hợp lệ")
            continue

        tool_name, arg_schema = TOOLS[choice]
        args = ask_args(arg_schema)
        try:
            result = call_remote_tool(tool_name, args)
            print(f"Calling with tool: {tool_name} and args: {args}")
            print("\n[Kết quả]")
            print(result)
        except Exception as e:
            print(f"[Lỗi] {e}")


if __name__ == "__main__":
    main()