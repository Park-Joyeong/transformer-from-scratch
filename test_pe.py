# test_pe.py
import torch
from model.positional_encoding import PositionalEncoding

if __name__ == "__main__":
    # batch: 문장 개수
    # seq_len: 하나의 문장을 몇개의 토큰으로 표현할지
    # d_model: 임베딩 차원(몇개의 차원으로 토큰을 표현할지)
    batch, seq_len, d_model = 2, 10, 16
    x = torch.zeros(batch, seq_len, d_model)

    pe = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
    out = pe(x)

    print("out shape:", out.shape)  # (2, 10, 16)
    print("first token pe:", out[0, 0, :])  # pos=0의 pe 값 확인
    print("second token pe:", out[0, 1, :]) # pos=1의 pe 값 확인
