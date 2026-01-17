# test_attn.py
import torch
from model.attention import SelfAttention

if __name__ == "__main__":
    B, T, d_model = 2, 5, 16
    x = torch.randn(B, T, d_model)

    sa = SelfAttention(d_model=d_model, dropout=0.0)
    out, attn = sa(x)

    print("out shape:", out.shape)     # (2, 5, 16)
    print("attn shape:", attn.shape)   # (2, 1, 5, 5)
    print("attn row sum (should be 1):", attn[0, 0, 0].sum().item())
