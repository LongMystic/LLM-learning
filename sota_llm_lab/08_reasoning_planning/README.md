## 08_reasoning_planning – Reasoning & Planning với LLM

Mục tiêu:
- Hiểu các **pattern reasoning** phổ biến: Chain-of-Thought (CoT), self-consistency, Tree-of-Thought (ToT) ở mức đơn giản.
- Thử nghiệm **planning**: để LLM tự lên kế hoạch (plan) rồi thực thi (execute) thay vì trả lời một bước.
- So sánh nhanh: trả lời thường vs có reasoning / planning (dùng chính LLM hiện có qua Ollama Gateway).

### Các bước dự kiến

1. **08.1 – Chain-of-Thought cơ bản**
   - Viết prompt cho cùng một bài toán (logic, giải thích, step-by-step) với 2 chế độ:
     - Không CoT (trả lời trực tiếp).
     - CoT (ép LLM “nghĩ từng bước”, rồi trích ra kết luận).
   - Ghi lại vài ví dụ để cảm nhận sự khác nhau.

2. **08.2 – Self-reflection đơn giản**
   - Cho LLM tự **review câu trả lời của chính nó**:
     - Round 1: trả lời bình thường.
     - Round 2: “critique / self-review” dựa trên yêu cầu ban đầu.
     - Round 3: sửa câu trả lời theo critique.
   - So sánh output trước/sau self-refine.

3. **08.3 – Planner → Executor (mini planning)**
   - Thiết kế 1 prompt cho **planner**: nhận task lớn, trả về danh sách step con (JSON).
   - Executor: thực thi từng step (ở mức đơn giản: chỉ là các call LLM tuần tự, không cần tool).
   - Dùng ví dụ gần gũi: “viết outline bài blog”, “tóm tắt rồi viết email báo cáo”, ...

4. **08.4 – Kết nối với Agentic AI (module 03)**
   - Mapping khái niệm:
     - CoT / self-reflection ~ critic / self-correction trong multi-agent.
     - Planner → Executor ~ orchestrator + worker agents.
   - Ghi chú lại bài học để sau quay lại module 03 đào sâu hơn.

> Giống các module gần đây, code `.py` cho 08 sẽ được mô tả trong message (code block), bạn tự tạo file để luyện tay.

