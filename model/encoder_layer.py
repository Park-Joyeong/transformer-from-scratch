# model/encoder_layer.py
import torch
import torch.nn as nn
from model.multihead import MultiHeadAttention


class PositionwiseFFN(nn.Module):
    """
    Position-wise Feed Forward Network (논문 3.3)

    각 토큰 위치별로 "똑같은" 2층 MLP를 적용함.
    입력/출력 shape은 유지: (B, T, d_model) -> (B, T, d_model)

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Transformer EncoderLayer (논문 3.1)

    구성:
      1) Multi-Head Self-Attention
      2) Add & Norm (Residual + LayerNorm)
      3) Position-wise FFN
      4) Add & Norm (Residual + LayerNorm)

    입력/출력:
      x: (B, T, d_model)
      out: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # LayerNorm은 feature 차원(d_model)에 대해 정규화
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        mask (선택): attention에서 사용할 마스크
        """
        # 1) Multi-Head Self-Attention
        attn_out, attn = self.mha(x, mask=mask)  # attn_out: (B, T, d_model)

        # 2) Residual + LayerNorm (Add & Norm)
        #    논문 표현: LayerNorm(x + Sublayer(x))
        x = self.norm1(x + self.dropout1(attn_out))

        # 3) FFN
        ffn_out = self.ffn(x)  # (B, T, d_model)

        # 4) Residual + LayerNorm (Add & Norm)
        out = self.norm2(x + self.dropout2(ffn_out))

        return out, attn
