# model/attention.py
import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    - Q: (B, H, Tq, d_k)
    - K: (B, H, Tk, d_k)
    - V: (B, H, Tk, d_v)  (usually d_v == d_k)
    - mask (optional): broadcastable to (B, H, Tq, Tk)
      mask == 0 where you want to block attention (set score to -inf)
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        d_k = Q.size(-1)

        # scores: (B, H, Tq, Tk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask if provided
        if mask is not None:
            # mask shape should be broadcastable to scores
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # attn: (B, H, Tq, Tk)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # out: (B, H, Tq, d_v)
        out = torch.matmul(attn, V)
        return out, attn


class SelfAttention(nn.Module):
    """
    Self-Attention layer: produces Q, K, V from the same input x.
    Input:
      x: (B, T, d_model)
    Output:
      out: (B, T, d_model)  (after final linear projection)
      attn: (B, H, T, T) if multi-head wrapper uses this
    Note: This class is "single-head style" by default.
          We'll build Multi-Head Attention next by reshaping into heads.
    """

    def __init__(self, d_model: int, d_k: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model  # single head => d_k=d_model

        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)

        self.attn = ScaledDotProductAttention(dropout=dropout)

        # project back to d_model (for consistency)
        self.W_o = nn.Linear(self.d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (B, T, d_model)
        Q = self.W_q(x)  # (B, T, d_k)
        K = self.W_k(x)  # (B, T, d_k)
        V = self.W_v(x)  # (B, T, d_k)

        # scaled dot attention expects (B, H, Tq, d_k)
        # for single-head, H=1
        Q = Q.unsqueeze(1)  # (B, 1, T, d_k)
        K = K.unsqueeze(1)  # (B, 1, T, d_k)
        V = V.unsqueeze(1)  # (B, 1, T, d_k)

        out, attn = self.attn(Q, K, V, mask=mask)  # out: (B, 1, T, d_k)

        out = out.squeeze(1)  # (B, T, d_k)
        out = self.W_o(out)   # (B, T, d_model)
        return out, attn
