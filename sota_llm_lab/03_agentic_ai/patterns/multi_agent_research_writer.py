"""
03.1 – Mini multi-agent workflow: Research Agent + Writer Agent + Orchestrator
"""
import json
import requests
from typing import Dict, Any

OLLAMA_URL = "http://localhost:11434/api/generate"
MCP_SERVER_URL = "http://localhost:8081/call_tool"
MODEL_NAME = "llama3.2"


def call_remote_tool(tool: str, args: dict):
    resp = requests.post(MCP_SERVER_URL, json={"tool": tool, "args": args})
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(data.get("error"))
    return data.get("result")


def call_ollama(prompt: str, format_json: bool = False) -> str | dict:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "format": "json" if format_json else None,
            "stream": False,
        },
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    if format_json:
        return json.loads(raw)
    return raw


def research_agent(user_goal: str) -> str:
    """
    Agent chuyên tìm kiếm thông tin qua MCP tools.
    Không dùng LLM để chọn tool (model 3B quá yếu cho việc này).
    Thay vào đó: dùng LLM phân tích goal → xác định tool + args → gọi tool trực tiếp.
    Sau khi có observations, dùng LLM tổng hợp.
    """
    plan_prompt = f"""Phân tích yêu cầu sau và chọn các tool cần gọi.

Yêu cầu: {user_goal}

TOOLS có sẵn:
- search_files: liệt kê tên file. args: {{"root": "...", "pattern": "*.md"}}
- read_file: đọc nội dung 1 file. args: {{"path": "..."}}
- grep_code: tìm dòng chứa chuỗi. args: {{"path": "...", "pattern": "..."}}

Trả về JSON dạng: {{"steps": [{{"tool": "...", "args": {{...}}}}]}}
Tối đa 4 bước. Trả DUY NHẤT JSON."""

    try:
        plan = call_ollama(plan_prompt, format_json=True)
        steps = plan.get("steps", [])
    except Exception:
        steps = [{"tool": "search_files", "args": {"root": ".", "pattern": "*.md"}}]

    if not steps:
        steps = [{"tool": "search_files", "args": {"root": ".", "pattern": "*.md"}}]

    observations = []
    for i, step in enumerate(steps[:4]):
        tool = step.get("tool", "")
        args = step.get("args", {})
        if not tool or not args:
            continue
        print(f"  [Research] Bước {i+1}: {tool}({args})")
        try:
            obs = call_remote_tool(tool, args)
        except Exception as e:
            obs = f"[ERROR] {e}"
        obs_str = str(obs)[:1500]
        observations.append(f"Tool: {tool}({args})\nKết quả:\n{obs_str}")
        print(f"  [Research] -> {obs_str[:200]}...")

        if tool == "search_files" and isinstance(obs, list) and obs:
            for file_path in obs[:5]:
                print(f"  [Research] Đọc file: {file_path}")
                try:
                    content = call_remote_tool("read_file", {"path": file_path, "max_chars": 500})
                except Exception as e:
                    content = f"[ERROR] {e}"
                observations.append(f"Tool: read_file({file_path})\nKết quả:\n{str(content)[:500]}")
            break  # đã có danh sách file + nội dung, không cần thêm step

    if not observations:
        return "(Không thu thập được dữ liệu nào)"

    all_obs = "\n---\n".join(observations)
    summary_prompt = f"""Dựa vào dữ liệu thật dưới đây, hãy tóm tắt ngắn gọn.
KHÔNG BỊA thêm thông tin. Chỉ dùng dữ liệu bên dưới.

Yêu cầu của user: {user_goal}

Dữ liệu thu thập được:
{all_obs}

Tóm tắt:"""

    return call_ollama(summary_prompt, format_json=False)


def writer_agent(research_summary: str, user_goal: str) -> str:
    """Agent chuyên viết lại thành báo cáo cho user."""
    if not research_summary or research_summary.startswith("(Không"):
        return "Không có dữ liệu để viết báo cáo. Research agent không tìm được thông tin."

    prompt = f"""Bạn là writer agent. Viết báo cáo DỰA TRÊN dữ liệu bên dưới.
KHÔNG BỊA thêm số liệu, tên file, hay thông tin nào không có trong dữ liệu.

Yêu cầu của user: {user_goal}

Dữ liệu từ research agent:
{research_summary}

Viết báo cáo ngắn gọn, có cấu trúc. Chỉ trình bày những gì có trong dữ liệu trên."""

    return call_ollama(prompt, format_json=False)


def orchestrator_agent(user_goal: str) -> str:
    """Orchestrator: điều phối research_agent -> writer_agent."""
    print(f"[Orchestrator] Nhận yêu cầu: {user_goal}")
    print("[Orchestrator] Gửi subtask cho research_agent...")

    research_result = research_agent(user_goal)
    print(f"[Research Agent] Kết quả tóm tắt: \n{research_result[:1000]}...")

    print(f"[Orchestrator] Gửi kết quả cho writer_agent...")
    final_report = writer_agent(research_result, user_goal)
    print(f"[Writer Agent] Báo cáo cuối cùng:\n{final_report[:1000]}...")

    return final_report


def interactive_multi_agent():
    print("=== Multi-Agent System (Research + Writer) - gõ 'exit' để thoát ===")
    while True:
        goal = input("\nUser goal> ").strip()
        if goal.lower() in ("exit", "quit", "q"):
            break

        try:
            report = orchestrator_agent(goal)
            print("\n" + "=" * 30)
            print("[Final Report]")
            print("=" * 30)
            print(report)
            print("=" * 30)
        except Exception as e:
            print(f"[Lỗi] {e}")


if __name__ == "__main__":
    interactive_multi_agent()
