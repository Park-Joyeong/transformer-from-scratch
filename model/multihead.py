# model/multihead.py
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 모듈

    입력:
      x_q: (B, Tq, d_model)
      x_k: (B, Tk, d_model)
      x_v: (B, Tk, d_model)
      mask (선택): (B, H, Tq, Tk) 로 broadcast 가능

    출력:
      out:  (B, Tq, d_model)
      attn: (B, H, Tq, Tk)  -> 각 head의 attention weight
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        # d_model은 head 수로 나누어 떨어져야 함
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # 각 head의 차원

        # Q, K, V를 만들기 위한 선형 변환
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # head들을 다시 합친 뒤 사용하는 출력 projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, T, d_model) → (B, H, T, d_head)

        하나의 큰 벡터를 여러 head로 쪼개는 함수
        """
        B, T, _ = x.shape

        # (B, T, H, d_head)
        x = x.view(B, T, self.num_heads, self.d_head)

        # (B, H, T, d_head)  -> attention 계산 편하게 차원 변경
        x = x.transpose(1, 2)
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, H, T, d_head) → (B, T, d_model)

        여러 head 결과를 다시 하나로 합침
        """
        B, H, T, d_head = x.shape

        # (B, T, H, d_head)
        x = x.transpose(1, 2)

        # (B, T, d_model)
        x = x.contiguous().view(B, T, H * d_head)
        return x

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        # self-attention일 경우 Q, K, V 전부 같은 입력 사용
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_k

        # 1) Q, K, V 선형 변환
        Q = self.W_q(x_q)  # (B, Tq, d_model)
        K = self.W_k(x_k)  # (B, Tk, d_model)
        V = self.W_v(x_v)  # (B, Tk, d_model)

        # 2) head로 분리
        Q = self._split_heads(Q)  # (B, H, Tq, d_head)
        K = self._split_heads(K)  # (B, H, Tk, d_head)
        V = self._split_heads(V)  # (B, H, Tk, d_head)

        # 3) Scaled Dot-Product Attention
        # scores: (B, H, Tq, Tk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # mask 적용 (decoder에서 미래 토큰 가릴 때 사용)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmax → attention weight
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 값 벡터 가중합
        out_heads = torch.matmul(attn, V)  # (B, H, Tq, d_head)

        # 4) head 합치기
        out = self._combine_heads(out_heads)  # (B, Tq, d_model)

        # 5) 최종 선형 변환
        out = self.W_o(out)

        return out, attn
