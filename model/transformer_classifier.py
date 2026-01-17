# model/transformer_classifier.py
import torch
import torch.nn as nn

from model.positional_encoding import PositionalEncoding
from model.encoder import Encoder


class TransformerClassifier(nn.Module):
    """
    Encoder-only Transformer로 문장 분류하기

    흐름:
      token_ids -> Embedding -> PositionalEncoding
      -> EncoderStack
      -> Pooling(평균/CLS 등)
      -> Linear -> logits

    입력:
      input_ids: (B, T)  (정수 토큰 ID)
    출력:
      logits: (B, num_classes)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_len: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def make_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        padding 토큰은 attention에서 '보면 안 됨'
        input_ids: (B, T)
        return mask: (B, 1, 1, T) 형태 (broadcast용)
          - 1: attention 허용
          - 0: attention 차단
        """
        # pad_id 아닌 곳은 True(1), pad_id인 곳은 False(0)
        mask = (input_ids != self.pad_id).unsqueeze(1).unsqueeze(2)
        # shape: (B, 1, 1, T)
        return mask

    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: (B, T)
        """
        # 1) Embedding
        x = self.embedding(input_ids)  # (B, T, d_model)

        # 2) Positional Encoding
        x = self.pe(x)  # (B, T, d_model)

        # 3) Padding mask
        mask = self.make_padding_mask(input_ids)  # (B, 1, 1, T)

        # 4) Encoder Stack
        enc_out, attn_list = self.encoder(x, mask=mask)  # (B, T, d_model)

        # 5) Pooling (평균 풀링: pad 제외)
        # mask: (B, 1, 1, T) -> (B, T, 1)로 바꿔서 곱하기 쉬움
        token_mask = (input_ids != self.pad_id).unsqueeze(-1).float()  # (B, T, 1)

        # pad 제외한 토큰만 합
        summed = (enc_out * token_mask).sum(dim=1)  # (B, d_model)
        count = token_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = summed / count  # (B, d_model)

        # 6) 분류기
        logits = self.classifier(pooled)  # (B, num_classes)

        return logits, attn_list
