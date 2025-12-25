import torch
import torch.nn as nn

torch.manual_seed(0)

class TinyBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=128, max_seq_len=16, vocab_size=5000):
        super().__init__()

        # token + position embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # feed-forward (MLP)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        # token_ids: [batch, seq_len]
        batch_size, seq_len = token_ids.shape

        # 1 ) embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.tok_emb(token_ids) + self.pos_emb(positions) # [B, T, d_model]

        # 2 ) causal mask so tokens can't see the future
        # True = block, False = allow
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=token_ids.device), diagonal=1).bool()

        # 3 ) self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask) # [B, T, d_model]

        # 4 ) residual + norm
        x = self.ln1(x + attn_out)

        # 5 ) feed-forward
        ff_out = self.ff(x) # [B, T, d_model]
        
        # 6 ) residual + norm
        x = self.ln2(x + ff_out)

        return x # [B, T, d_model]

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

