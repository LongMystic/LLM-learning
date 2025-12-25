## 5. LLMOps & Serving (3–4 weeks)

### Goal
Serve models via an API with basic observability, caching, and controls.

### Learn
- High-level serving options (vLLM, TGI)
- FastAPI basics and streaming responses
- Observability concepts:
  - Structured logging
  - Latency metrics
  - Request sampling
- Caching and rate limiting

### Do
- Run a model server locally (vLLM or TGI).
- Expose it via FastAPI:
  - Chat endpoint
  - Streaming token responses
  - Pydantic request/response models
- Add observability:
  - Structured logs with trace IDs, latency, and user/task info
  - Basic latency histogram (e.g., via logs or a small metrics library)
- Add:
  - Redis cache for responses (e.g., semantic or key-based)
  - Simple rate limiting per user/IP

### Done when
- You can send requests to your FastAPI service and see logs with trace IDs and latency.
- Cache hits reduce latency in repeated queries.

