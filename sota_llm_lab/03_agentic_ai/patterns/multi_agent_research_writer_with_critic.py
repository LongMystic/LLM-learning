"""
03.3 – Multi-agent với Critic (feedback & self-reflection)
- Tái sử dụng research_agent, writer_agent, call_ollama từ multi_agent_research_writer.py
- Thêm critic_agent: review báo cáo so với yêu cầu ban đầu và đề xuất chỉnh sửa.
- Vòng lặp tối đa N lần tự sửa trước khi trả kết quả cho user.
"""
import json
from typing import Literal

from multi_agent_research_writer import (
    research_agent,
    writer_agent,
    call_ollama,
)

MAX_CRITIC_ROUNDS = 3  # số vòng self-refine tối đa


def critic_agent(user_goal: str, draft_report: str) -> dict:
    """
    Critic agent:
    - Đọc yêu cầu user + bản báo cáo.
    - Quyết định:
      + "approve": báo cáo đã đáp ứng yêu cầu.
      + "revise": cần chỉnh sửa, kèm bản báo cáo đã chỉnh sửa và comment ngắn.
    Trả về JSON: {"action": "approve"} hoặc
                  {"action": "revise", "revised_report": "...", "comment": "..."}
    """
    prompt = f"""Bạn là critic agent.

Yêu cầu ban đầu của user:
\"\"\"{user_goal}\"\"\"

Bản báo cáo hiện tại:
\"\"\"{draft_report}\"\"\"

Nhiệm vụ:
1. Kiểm tra báo cáo đã:
   - Trả lời trúng trọng tâm yêu cầu chưa.
   - Có dùng dữ liệu/quan sát từ research (nếu có) một cách hợp lý không.
   - Có thiếu ý quan trọng nào không.
2. Nếu báo cáo đã ổn: trả về JSON
   {{"action": "approve"}}
3. Nếu cần chỉnh sửa:
   - Viết lại báo cáo tốt hơn trong field "revised_report".
   - Thêm một comment ngắn trong field "comment" giải thích điểm đã cải thiện.
   - Trả về JSON:
   {{
     "action": "revise",
     "revised_report": "...",
     "comment": "..."
   }}

CHỈ TRẢ VỀ DUY NHẤT 1 JSON HỢP LỆ, KHÔNG THÊM TEXT NÀO KHÁC.
"""

    resp = call_ollama(prompt, format_json=True)
    # resp ở đây đã là dict (vì ta dùng format_json=True)
    action = resp.get("action")
    if action not in ("approve", "revise"):
        # fallback an toàn
        return {"action": "approve"}
    return resp


def orchestrator_with_critic(user_goal: str) -> str:
    """
    Orchestrator:
    - gọi research_agent → writer_agent → critic_agent (tối đa N vòng).
    """
    print(f"[Orchestrator] Nhận yêu cầu: {user_goal}")
    print("[Orchestrator] Gửi subtask cho research_agent...")

    research_result = research_agent(user_goal)
    print(f"[Research Agent] Kết quả tóm tắt (preview):\n{research_result[:400]}...\n")

    # Vòng writer + critic
    current_report = writer_agent(research_result, user_goal)
    print("[Writer Agent] Bản nháp đầu tiên (preview):")
    print(current_report[:400], "...\n")

    for round_idx in range(1, MAX_CRITIC_ROUNDS + 1):
        print(f"[Critic Agent] Vòng review {round_idx}/{MAX_CRITIC_ROUNDS}...")
        review = critic_agent(user_goal, current_report)
        action = review.get("action")

        if action == "approve":
            print("[Critic Agent] Báo cáo đã đạt yêu cầu, chấp nhận.")
            break

        if action == "revise":
            revised = review.get("revised_report") or current_report
            comment = review.get("comment", "")
            print("[Critic Agent] Đề xuất chỉnh sửa:")
            if comment:
                print("  Comment:", comment)
            print("[Critic Agent] Áp dụng bản revised_report (preview):")
            print(revised[:400], "...\n")
            current_report = revised
        else:
            # fallback nếu JSON lạ
            print("[Critic Agent] Phản hồi không hợp lệ, giữ nguyên báo cáo hiện tại.")
            break

    return current_report


def interactive_multi_agent_with_critic():
    print("=== Multi-Agent (Research + Writer + Critic) – gõ 'exit' để thoát ===")
    while True:
        goal = input("\nUser goal> ").strip()
        if goal.lower() in ("exit", "quit", "q"):
            break

        try:
            final_report = orchestrator_with_critic(goal)
            print("\n" + "=" * 40)
            print("[Final Report]")
            print("=" * 40)
            print(final_report)
            print("=" * 40)
        except Exception as e:
            print(f"[Lỗi] {e}")


if __name__ == "__main__":
    interactive_multi_agent_with_critic()