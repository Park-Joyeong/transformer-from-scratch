# test_encoder_layer.py
import torch
from model.encoder_layer import EncoderLayer

if __name__ == "__main__":
    B, T, d_model = 2, 5, 16
    num_heads = 4
    d_ff = 64

    x = torch.randn(B, T, d_model)

    layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)
    out, attn = layer(x)

    print("out shape:", out.shape)     # (2, 5, 16)
    print("attn shape:", attn.shape)   # (2, 4, 5, 5)
    print("attn row sum:", attn[0, 0, 0].sum().item())  # ~1.0
