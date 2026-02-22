import json
import requests

OLLAMA_URL = 'http://localhost:11434/api/generate'
MCP_SERVER_URL = 'http://localhost:8081/call_tool'
MODEL_NAME = 'llama3.2'

def call_remote_tool(tool: str, args: dict):
    resp = requests.post(
        MCP_SERVER_URL, 
        json={
            'tool': tool,
            'args': args
        }
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok", False) and data.get('result') is not None:
        raise RuntimeError(f"Tool error: {data.get('error')}")
    return data.get("result")
    # return data

def plan_with_ollama(user_query: str) -> dict:
    """
        Gọi Ollama, yêu cầu nó trả về DUY NHẤT 1 JSON:
        {"tool": "...", "args": {...}}
    """
    tools_description = """
        Bạn có các tool sau:

        1) echo
        - Mô tả: Lặp lại text
        - Cách gọi: {"tool": "echo", "args": {"text": "nội dung text"}}

        2) read_file
        - Mô tả: Đọc nội dung 1 file text
        - Cách gọi: {"tool": "read_file", "args": {"path": "đường/dẫn/file", "max_chars": 2000}}

        3) search_files
        - Mô tả: Tìm file theo pattern trong 1 thư mục
        - Cách gọi: {"tool": "search_files", "args": {"root": "đường/dẫn/thư mục", "pattern": "*.py", "max_results": 50}}

        5) grep_code
        - Mô tả: Tìm dòng chứa pattern trong file/thư mục code.
        - Cách gọi: {"tool": "grep_code", "args": {"path": "đường/dẫn/file hoặc thư mục", "pattern": "chuỗi tìm", "max_lines": 50}}
        6) read_log
        - Mô tả: Đọc N dòng cuối của file.
        - Cách gọi: {"tool": "read_log", "args": {"path": "đường/dẫn/file", "last_n_lines": 10}}
        7) query_sqlite
        - Mô tả: Chạy câu SELECT trên file SQLite (.db).
        - Cách gọi: {"tool": "query_sqlite", "args": {"db_path": "đường/dẫn/file.db", "query": "SELECT ... FROM ..."}}

        Quy tắc chọn tool:
        - Nếu người dùng muốn xem nội dung của một file (các từ khóa: "đọc file", "mở file", "nội dung file", "show file", "xem file") thì BẮT BUỘC dùng tool "read_file".
        - Nếu người dùng chỉ muốn bạn lặp lại một câu, hoặc in ra đúng câu họ đưa (các từ khóa: "echo", "lặp lại", "nhắc lại", "repeat") thì dùng tool "echo".
        - Nếu người dùng muốn TÌM DÒNG trong nội dung file hoặc thư mục code (các từ khóa: "tìm dòng", "grep", "pattern trong file", "chứa chuỗi", "search text", "search content"), thì BẮT BUỘC dùng tool "grep_code", KHÔNG dùng "search_files".
        - Nếu người dùng muốn LIỆT KÊ / TÌM FILE (các từ khóa: "liệt kê file", "danh sách file", "tìm file", "list file", "file nào tồn tại"), thì dùng tool "search_files", KHÔNG dùng "grep_code".

        YÊU CẦU QUAN TRỌNG:
        - Chỉ trả về DUY NHẤT 1 JSON, không thêm giải thích, không thêm text thừa
    """

    prompt = f"""
        Bạn là bộ lập kế hoạch tool cho một hệ MCP.

        User hỏi:
        \"\"\"{user_query}\"\"\"

        {tools_description}

        Hãy chọn tool phù hợp nhất và điền args đúng.
        Trả về DUY NHẤT một JSON hợp lệ, ví dụ:
        {{"tool": "read_file", "args": {{"path": "README.md"}}}}
    """

    resp = requests.post(
        OLLAMA_URL,
        json={
            'model': MODEL_NAME,
            'prompt': prompt,
            'stream': False
        },
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "").strip()

    # Cố gắng parse JSON
    try:
        plan = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Ollama trả về không phải JSON hợp lệ: {text}")
    
    if "tool" not in plan or "args" not in plan:
        raise ValueError(f"Plan thiếu 'tool' hoặc 'args': {plan}")

    return plan


def interactive_loop():
    print("=== Ollama MCP Planner (gõ 'exit' để thoát) ===")
    while True:
        user_query = input("\nUser> ").strip()
        if user_query.lower() in ["exit", "quit", "q"]:
            break

        try:
            plan = plan_with_ollama(user_query)
            print(f"[Planner] Kế hoạch: {plan}")

            result = call_remote_tool(plan["tool"], plan["args"])
            print("[Tool result]")
            print(result)
        except Exception as e:
            print(f"[Lỗi] {e}")


if __name__ == "__main__":
    interactive_loop()