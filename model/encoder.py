# model/encoder.py
import torch
import torch.nn as nn
from model.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    Transformer Encoder (논문 3.1)
    EncoderLayer를 N개 쌓은 구조

    입력/출력:
      x: (B, T, d_model)
      out: (B, T, d_model)
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        mask (선택): (B, H, T, T)로 broadcast 가능하거나 (B, 1, 1, T) 형태 등도 가능
        """
        attn_list = []

        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attn_list.append(attn)

        return x, attn_list
