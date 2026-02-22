import json
import requests

OLLAMA_URL = 'http://localhost:11434/api/generate'
MCP_SERVER_URL = 'http://localhost:8081/call_tool'
MODEL_NAME = 'llama3.2'


def call_remote_tool(tool: str, args: dict):
    resp = requests.post(MCP_SERVER_URL, json={'tool': tool, 'args': args})
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(data.get("error"))
    return data.get("result")


TOOLS_CATALOG = """TOOLS (chỉ dùng đúng tên và args dưới đây):
- echo: lặp lại text. args: {"text": "..."}
- read_file: đọc nội dung file. args: {"path": "...", "max_chars": 2000}
- search_files: liệt kê TÊN file theo đuôi. args: {"root": "...", "pattern": "*.py"}
- grep_code: tìm các DÒNG chứa chuỗi trong NỘI DUNG file/thư mục (giống grep). args: {"path": "...", "pattern": "...", "max_lines": 30}
- read_log: đọc N dòng cuối file. args: {"path": "...", "last_n_lines": 50}
- query_sqlite: chạy SELECT trên SQLite. args: {"db_path": "...", "query": "SELECT ..."}"""


def summarize_with_ollama(user_goal: str, history: list) -> str:
    """Gọi Ollama KHÔNG ép JSON để tổng hợp observations thành câu trả lời tự nhiên."""
    obs_text = ""
    for i, step in enumerate(history, 1):
        obs_text += f"\nBước {i}: gọi {step['tool']}({step['args']})\n"
        obs_text += f"Kết quả:\n{step['observation'][:1000]}\n"

    prompt = f"""User yêu cầu: {user_goal}

Dưới đây là dữ liệu thật thu được từ các tool:
{obs_text}

Hãy tổng hợp dữ liệu trên thành câu trả lời đầy đủ, rõ ràng cho user. Liệt kê chi tiết kết quả, không bỏ sót."""

    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def ask_ollama_for_action(user_goal: str, history: list) -> dict:
    """
    history: list các bước trước, mỗi bước dạng
    {"tool": ..., "args": ..., "observation": "..."}
    """
    history_text = ""
    for i, step in enumerate(history, 1):
        history_text += f"\nBước {i}:\n"
        history_text += f" - tool đã gọi: {step['tool']}\n"
        history_text += f" - args: {step['args']}\n"
        history_text += f" - observation: {step['observation'][:500]}\n"

    prompt = f"""Bạn là agent. Trả về DUY NHẤT 1 JSON.

TOOLS:
- read_file: đọc nội dung 1 FILE (không phải thư mục). args: {{"path": "file.py"}}
- search_files: liệt kê tên file theo đuôi trong thư mục. args: {{"root": "thư_mục", "pattern": "*.py"}}
- grep_code: tìm DÒNG chứa chuỗi trong file/thư mục (giống lệnh grep). args: {{"path": "thư_mục_hoặc_file", "pattern": "chuỗi", "max_lines": 30}}
- echo: lặp lại text. args: {{"text": "..."}}
- read_log: đọc dòng cuối file. args: {{"path": "file.log", "last_n_lines": 50}}
- query_sqlite: chạy SELECT trên SQLite. args: {{"db_path": "file.db", "query": "SELECT ..."}}

VÍ DỤ HOÀN CHỈNH:
User yêu cầu: "Tìm các dòng chứa import trong thư mục src rồi tóm tắt"
Bước 1 (chưa có observation): {{"action": "call_tool", "tool": "grep_code", "args": {{"path": "src", "pattern": "import", "max_lines": 30}}}}
Bước 2 (đã có observation chứa kết quả grep): {{"action": "final_answer", "answer": "Tìm thấy 5 dòng import: ..."}}

User yêu cầu: "Đọc file README.md"
Bước 1: {{"action": "call_tool", "tool": "read_file", "args": {{"path": "README.md"}}}}
Bước 2: {{"action": "final_answer", "answer": "Nội dung file README.md: ..."}}

LƯU Ý:
- read_file CHỈ có args "path" (và max_chars), dùng cho 1 FILE cụ thể. KHÔNG có root hay pattern.
- Liệt kê file theo đuôi (*.md, *.py) -> dùng search_files với root + pattern. Sau khi search_files trả danh sách, trả NGAY final_answer, không gọi thêm tool.
- Muốn tìm NỘI DUNG bên trong file -> dùng grep_code. Pattern PHẢI đúng chuỗi user yêu cầu.

BÂY GIỜ:
User yêu cầu: {user_goal}

Các bước đã làm:
{history_text if history_text else "(chưa gọi tool nào)"}

Trả về 1 JSON:
- Gọi tool: {{"action": "call_tool", "tool": "...", "args": {{...}}}}
- Trả lời cuối: {{"action": "final_answer", "answer": "..."}}"""

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "").strip()

    # format="json" vẫn trả string JSON, cần parse
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Ollama trả về không phải JSON hợp lệ:\n{raw}")

    action = plan.get("action")
    if action not in ("call_tool", "final_answer"):
        if plan.get("tool") and "args" in plan:
            plan["action"] = "call_tool"
        else:
            raise ValueError(f"action không hợp lệ (phải là call_tool hoặc final_answer): {plan}")

    return plan


def interactive_agent():
    print("=== Ollama ReAct Agent (gõ 'exit' để thoát) ===")
    while True:
        goal = input("\nUser goal> ").strip()
        if goal.lower() in ("exit", "quit", "q"):
            break

        history = []
        MAX_STEPS = 5
        MAX_RETRIES = 2

        for step_idx in range(MAX_STEPS):
            plan = None
            for retry in range(MAX_RETRIES + 1):
                try:
                    plan = ask_ollama_for_action(goal, history)
                    break
                except ValueError as e:
                    print(f"[Retry {retry + 1}/{MAX_RETRIES}] {e}")
                    if retry == MAX_RETRIES:
                        print("[Lỗi] Không parse được sau nhiều lần retry.")
                        break

            if plan is None:
                break

            action = plan.get("action")

            if action == "final_answer":
                if history:
                    summary = summarize_with_ollama(goal, history)
                    print("\n[Final answer]")
                    print(summary)
                else:
                    print("\n[Final answer]")
                    print(plan.get("answer"))
                break

            if action == "call_tool":
                tool = plan.get("tool")
                args = dict(plan.get("args") or {})
                if tool == "read_file" and ("root" in args or "pattern" in args):
                    tool, args = "search_files", {"root": args.get("root", args.get("path", ".")), "pattern": args.get("pattern", "*.md"), "max_results": args.get("max_results", 50)}
                print(f"\n[Agent] Gọi tool: {tool} với args: {args}")
                try:
                    observation = call_remote_tool(tool, args)
                except Exception as e:
                    observation = f"[ERROR] {e}"
                history.append({"tool": tool, "args": args, "observation": str(observation)})
                print("[Observation]")
                print(str(observation)[:500])


if __name__ == "__main__":
    interactive_agent()
