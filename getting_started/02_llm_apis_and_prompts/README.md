## 2. Core LLM Usage (2–3 weeks)

### Goal
Be able to call LLM APIs and design effective prompts.

### Learn
- Prompt patterns:
  - Zero-shot and few-shot
  - Chain-of-thought
  - Role–task–context–style structure
  - JSON / structured output prompts
- Decoding parameters:
  - `temperature`, `top_p`, `top_k`, `max_tokens`

### Do
- Set up an API provider (OpenAI or Anthropic) and a local model (`ollama` or `llama.cpp`).
- Write a small Python wrapper function to send prompts and get responses.
- Design prompts for:
  - Classification
  - Extraction (return JSON)
  - Reasoning with chain-of-thought
- Run a small experiment:
  - Take ~20 prompts and vary `temperature` / `top_p`
  - Log outputs to a CSV and visually inspect differences

### Done when
- You can pick prompt patterns and decoding settings for a new task and justify your choices.
- You have a simple script that can query both a hosted LLM and a local model.

