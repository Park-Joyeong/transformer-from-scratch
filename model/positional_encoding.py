# model/positional_encoding.py
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Transformer는 RNN/CNN처럼 '순서'를 자연스럽게 처리하지 않기 때문에
    입력 임베딩에 위치 정보를 더해줘야 한다. (논문 3.5 Positional Encoding :contentReference[oaicite:1]{index=1})

    논문에서 제안한 사인/코사인 방식:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model/2,)
        # 10000^(2i/d_model) 를 exp로 계산: exp( log(10000) * (2i/d_model) )
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # 짝수 인덱스(0,2,4...)에는 sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 홀수 인덱스(1,3,5...)에는 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model)로 만들어 batch 차원에 broadcast되게 함
        pe = pe.unsqueeze(0)

        # 학습 파라미터가 아니라 '고정 값'으로 저장
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        return: (batch, seq_len, d_model)

        입력 임베딩 x에 위치 인코딩을 더해준다.
        """
        seq_len = x.size(1)

        # pe[:, :seq_len, :] -> (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)
