# test_encoder_stack.py
import torch
from model.transformer_classifier import TransformerClassifier

if __name__ == "__main__":
    # 가짜 설정(작게)
    vocab_size = 100
    num_classes = 3
    pad_id = 0

    B, T = 2, 6
    max_len = 32

    model = TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_len=max_len,
        num_layers=2,
        d_model=16,
        num_heads=4,
        d_ff=64,
        dropout=0.0,
        pad_id=pad_id,
    )

    # 가짜 토큰 ID (0은 pad)
    input_ids = torch.tensor([
        [5, 7, 9, 2, 0, 0],
        [4, 1, 3, 8, 6, 0],
    ])

    logits, attn_list = model(input_ids)

    print("logits shape:", logits.shape)  # (2, 3)
    print("num layers attn:", len(attn_list))  # 2
    print("attn shape layer0:", attn_list[0].shape)  # (2, 4, 6, 6)
