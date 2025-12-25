## 1. Foundations (2–3 weeks)

### Goal
Understand transformer basics and be comfortable with the Python ML stack.

### Learn
- Python ML: `numpy`, `pandas`, `pyarrow`, basic GPU concepts.
- Transformer components: embeddings, self-attention, positional encodings, MLP, residuals, layernorm.
- Read/skim:
  - “Attention Is All You Need” (focus on attention)
  - Karpathy’s minGPT / nanoGPT explanation

### Do
- Install:
  - Python 3.10+, virtualenv
  - `torch`, `transformers`, `accelerate`, `datasets`, `einops`, `numpy`, `pandas`, `pyarrow`, `tiktoken`
- Implement a tiny transformer block forward pass on dummy data (batch, seq_len, d_model).
- Run a tiny `minGPT`/`nanoGPT` example on small text and watch the loss curve.

### Done when
- You can explain attention, masking, positional encodings in your own words.
- You have a small script that runs a transformer block forward pass without errors.


