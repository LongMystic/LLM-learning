## Explanation about code – TinyBlock (Foundations)

This file explains your `TinyBlock` step by step from a data engineer / beginner-ML point of view.

### 1. Big picture: what does `TinyBlock` do?

`TinyBlock` is one **transformer block**:

- Input: token IDs shaped like `[batch, seq_len]`, e.g. 2 sentences of 8 tokens each.
- Output: vectors shaped like `[batch, seq_len, d_model]`, e.g. each token becomes a 64‑dim vector.

It does **no training**, only a **forward pass** (\"take data → apply layers → get result\").

### 2. Class definition and parameters

```python
class TinyBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=128, max_seq_len=16, vocab_size=5000):
        super().__init__()
```

- `nn.Module`: base class for all neural network components in PyTorch.
- `d_model=64`: size of each token embedding vector (number of features per token).
- `n_heads=4`: how many *attention heads* inside multi‑head attention.
- `d_ff=128`: hidden size of the feed‑forward (MLP) layer; usually > `d_model`.
- `max_seq_len=16`: maximum number of tokens per sequence for position embeddings.
- `vocab_size=5000`: number of different token IDs we can represent (size of vocabulary).

These are **design choices**, not magic constants.

### 3. Embedding layers

```python
        # token + position embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
```

- `tok_emb`:
  - Maps **token IDs → vectors** of length `d_model`.
  - Think of a big lookup table of shape `[vocab_size, d_model]`.
- `pos_emb`:
  - Maps **position index → vector** of length `d_model`.
  - Adds information about where in the sequence the token is (0th, 1st, ...).

Later we **add** these two vectors to get \"token meaning + position\".

### 4. Self‑attention layer

```python
        # self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
```

- `MultiheadAttention`:
  - For each token, it learns **how much to attend to other tokens** in the same sequence.
  - `d_model` = size of vectors going in/out.
  - `n_heads` = how many parallel attention heads.
  - `batch_first=True` means our tensors are shaped `[batch, seq_len, d_model]`.

### 5. Feed‑forward (MLP) layer

```python
        # feed-forward (MLP)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
```

- This is applied **independently to each token** after attention.
- Two linear layers with a GELU activation in between:
  - First expands to `d_ff` (e.g., 128),
  - Then projects back to `d_model` (64).

You can think of it as \"local thinking\" per token after it has combined context via attention.

### 6. Layer norms

```python
        # layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
```

- `LayerNorm` keeps activations in a stable range.
- There are two norms:
  - One around the attention part,
  - One around the MLP part.

This is part of the standard transformer design to make training easier and more stable.

### 7. Forward pass – step by step

```python
    def forward(self, token_ids):
        # token_ids: [batch, seq_len]
        batch_size, seq_len = token_ids.shape
```

`token_ids` are integers like:

- Example shape: `[2, 8]`
- Example values: `[[12, 45, 7, ...], [3, 999, 2, ...]]`

#### 7.1 Embeddings

```python
        # 1 ) embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.tok_emb(token_ids) + self.pos_emb(positions) # [B, T, d_model]
```

- `positions`: tensor `[0, 1, 2, ..., seq_len-1]`.
- `self.tok_emb(token_ids)`:
  - Shape: `[batch, seq_len, d_model]`.
- `self.pos_emb(positions)`:
  - Shape: `[seq_len, d_model]`, and PyTorch broadcasts it over the batch.
- Sum = **\"token meaning + position\"** for each token.

Now `x` has shape `[batch, seq_len, d_model]`.

#### 7.2 Causal mask

```python
        # 2 ) causal mask so tokens can't see the future
        # True = block, False = allow
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=token_ids.device), diagonal=1).bool()
```

- `torch.ones(seq_len, seq_len)` → square matrix of 1s.
- `torch.triu(..., diagonal=1)` → keeps only the upper triangle **above** the diagonal.
  - Positions where the **column index > row index** are set to 1 (future positions).
- `.bool()` → convert 1s to `True`, others to `False`.

Meaning:

- For a given token at position `i`, we **block attention to positions `j > i`**.
- This enforces a **causal (autoregressive) mask**: no looking into the future.

#### 7.3 Self‑attention

```python
        # 3 ) self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask) # [B, T, d_model]
```

- We pass `x` as:
  - `query`, `key`, and `value` (standard self‑attention).
- `attn_mask=causal_mask`:
  - Tells attention which positions to block (the future).
- Output `attn_out` has the same shape as `x`: `[batch, seq_len, d_model]`.

Conceptually:

- Each token now has a vector that is a **weighted combination** of other tokens in the sequence.

#### 7.4 Residual + LayerNorm (after attention)

```python
        # 4 ) residual + norm
        x = self.ln1(x + attn_out)
```

- Residual: `x + attn_out`
  - Keeps original information and adds the new attention information.
- `self.ln1(...)`:
  - Normalizes to keep values from exploding or vanishing.

This is written as \"Add & Norm\" in many transformer diagrams.

#### 7.5 Feed-forward

```python
        # 5 ) feed-forward
        ff_out = self.ff(x) # [B, T, d_model]
```

- Applies the MLP per token, shape stays `[batch, seq_len, d_model]`.

#### 7.6 Residual + LayerNorm (after MLP)

```python
        # 6 ) residual + norm
        x = self.ln2(x + ff_out)

        return x # [B, T, d_model]
```

- Again, residual (`x + ff_out`) + layer norm.
- Returns the final transformed representation.

### 8. Main block (how you run it)

```python
if __name__ == "__main__":
    # dummy token IDs
    batch = 2
    seq_len = 8
    vocab_size = 5000

    token_ids = torch.randint(0, vocab_size, (batch, seq_len))
    model = TinyBlock(d_model=64, n_heads=4, d_ff=128, max_seq_len=seq_len, vocab_size=vocab_size)

    out = model(token_ids)
    print("Input shape: ", token_ids.shape) # [2, 8]
    print("Output shape: ", out.shape) # [2, 8, 64]
```

- `torch.randint(0, vocab_size, (batch, seq_len))`:
  - Creates random token IDs to simulate input.
- `model = TinyBlock(...)`:
  - Instantiates the block with your chosen hyperparameters.
- `out = model(token_ids)`:
  - Calls the `forward` method.
- Prints:
  - Input shape `[2, 8]` (2 sequences of 8 token IDs).
  - Output shape `[2, 8, 64]` (each token → 64‑dim vector).

### 9. What to learn around this (as a DE with little ML)

Before going deeper, it helps to:

1. Do a short **PyTorch basics** tutorial:
   - Tensors, shapes, `nn.Linear`, `nn.Embedding`, `nn.Module`, `forward`.
2. Get an intuition for **self‑attention**:
   - High level: \"each token chooses which other tokens to pay attention to\".
3. Be comfortable reading tensor shapes:
   - `[batch, seq_len, d_model]` and how operations preserve or change shapes.

You do **not** need full math of transformers yet; running and understanding this block step by step is exactly the right level.


