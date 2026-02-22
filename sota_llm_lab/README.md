## Roadmap SOTA LLM (Local / Self-host)

Mục tiêu: nắm vững các chủ đề LLM hiện đại và **tự tay build** các hệ thống chạy local/self-host, bắt đầu từ **MCP → Agents → Agentic AI**, sau đó mở rộng sang RAG nâng cao, fine-tuning, serving/LLMOps, reasoning và multimodal.

### Tổng quan các chặng

- **00_env_setup**: Chuẩn bị môi trường, môi trường ảo, GPU (nếu có), cấu trúc repo và các công cụ cơ bản.
- **01_mcp**: Hiểu và triển khai Model Context Protocol (MCP), tự viết server & client để kết nối LLM với công cụ/dịch vụ local.
- **02_agents**: Xây dựng agent đơn và multi-agent, dùng tool calling, phản hồi có trạng thái (stateful).
- **03_agentic_ai**: Thiết kế hệ thống Agentic AI hoàn chỉnh (workflow, planning, memory, tool orchestration).
- **04_rag_advanced**: RAG nâng cao (hybrid search, reranking, chunking thông minh, query rewriting).
- **05_finetuning_peft**: Fine-tune LLM bằng PEFT (LoRA/QLoRA) cho domain cụ thể.
- **06_serving_llmops**: Tối ưu serving & LLMOps (vLLM, quantization, deploy local/self-host).
- **07_reasoning_planning**: Kỹ thuật reasoning (CoT, ToT, self-reflection) và đánh giá.
- **08_multimodal_moe**: Multimodal, MoE, và các kiến trúc mới.

Bạn có thể đi **tuần tự từ 00 → 08**, hoặc tập trung sâu từng module.

---

### Gợi ý thứ tự học

1. **00_env_setup** (1–2 ngày)
2. **01_mcp** (3–5 ngày)
3. **02_agents** (5–7 ngày)
4. **03_agentic_ai** (7–10 ngày)
5. Sau đó chọn tiếp: **04** (RAG) → **05** (fine-tuning) → **06** (serving) → **07–08** (reasoning, multimodal, MoE).

---

### Checkpoint gợi ý

- Sau **01_mcp**: tự viết được ít nhất 1 MCP server kết nối tới 1 dịch vụ local (VD: đọc file, search, gọi API).
- Sau **02_agents**: có 1 agent CLI hoặc web nhỏ tự chọn tool, solve tasks nhiều bước.
- Sau **03_agentic_ai**: có 1 mini-project agentic hoàn chỉnh (VD: “research assistant”, “code mentor”, “data analyst assistant”).

