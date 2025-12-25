## 6. Orchestration & Agents (2–3 weeks)

### Goal
Know when and how to use orchestration frameworks and basic agent patterns.

### Learn
- LangChain / LlamaIndex basics:
  - Chains
  - Tools
  - Agents
- When plain Python is simpler than using a framework

### Do
- Build a simple chain in LangChain or LlamaIndex:
  - Retrieval step + LLM call
- Add a tool-calling agent:
  - One or two tools (e.g., calculator, web search mock, database query mock)
  - Timeouts and retries with backoff
- Add simple prompt-injection defenses:
  - Input filters
  - Output allowlists or regex checks

### Done when
- You can implement a small, agent-style workflow with tools and explain its limitations.
- You can articulate when to use a framework vs pure Python.

