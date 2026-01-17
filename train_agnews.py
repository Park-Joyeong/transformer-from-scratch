# train_agnews.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataset_agnews import make_dataloaders
from model.transformer_classifier import TransformerClassifier


# -------------------------
# 설정(필요하면 여기만 바꾸면 됨)
# -------------------------
@dataclass
class Config:
    # data
    max_len: int = 128
    max_vocab_size: int = 20000
    min_freq: int = 2
    batch_size: int = 32

    # model
    num_layers: int = 2
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1

    # train
    lr: float = 3e-4
    epochs: int = 3

    # save
    out_dir: str = "runs"


def get_device() -> torch.device:
    # Colab이면 cuda, Mac M2면 mps 가능, 아니면 cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits, _ = model(input_ids)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def train_one_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for input_ids, labels in pbar:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def main():
    cfg = Config()
    device = get_device()
    print("device:", device)

    # 1) Data
    vocab, train_loader, test_loader = make_dataloaders(
        max_len=cfg.max_len,
        max_vocab_size=cfg.max_vocab_size,
        min_freq=cfg.min_freq,
        batch_size=cfg.batch_size,
        num_workers=0,
    )
    num_classes = 4  # AG_NEWS는 4-class

    # 2) Model
    model = TransformerClassifier(
        vocab_size=len(vocab.itos),
        num_classes=num_classes,
        max_len=cfg.max_len,
        num_layers=cfg.num_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_id=vocab.pad_id,
    ).to(device)

    # 3) Optimizer
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # 4) Save dir
    run_name = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # config 저장(README에 복붙하기 좋게)
    with open(os.path.join(out_dir, "config.txt"), "w", encoding="utf-8") as f:
        for k, v in cfg.__dict__.items():
            f.write(f"{k}={v}\n")
        f.write(f"vocab_size={len(vocab.itos)}\n")

    print(f"run_dir: {out_dir}")

    # 5) Train loop
    best_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, device)

        print(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={val_loss:.4f}, test_acc={val_acc:.4f}"
        )

        # best 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    print("best_test_acc:", best_acc)


if __name__ == "__main__":
    main()
