# test_mha.py
import torch
from model.multihead import MultiHeadAttention

if __name__ == "__main__":
    B, T, d_model, H = 2, 5, 16, 4
    x = torch.randn(B, T, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=H, dropout=0.0)
    out, attn = mha(x)  # self-attention

    print("출력 shape:", out.shape)      # (2, 5, 16)
    print("attention shape:", attn.shape) # (2, 4, 5, 5)

    # attention weight 한 줄의 합은 거의 1이어야 함
    print("attention row sum:", attn[0, 0, 0].sum().item())
