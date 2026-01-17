# dataset_agnews.py
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# -------------------------
# 1) 아주 단순한 토크나이저
# -------------------------
def simple_tokenize(text: str) -> List[str]:
    # 소문자 + 공백 기준 split (가장 안정적)
    return text.lower().strip().split()


# -------------------------
# 2) vocab 만들기
# -------------------------
@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int = 0
    unk_id: int = 1

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]


def build_vocab(
    texts: List[str],
    max_vocab_size: int = 20000,
    min_freq: int = 2,
) -> Vocab:
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    # special tokens
    itos = ["<pad>", "<unk>"]
    stoi = {w: i for i, w in enumerate(itos)}

    # 빈도 높은 순으로 단어 추가
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if word in stoi:
            continue
        if len(itos) >= max_vocab_size:
            break
        stoi[word] = len(itos)
        itos.append(word)

    return Vocab(stoi=stoi, itos=itos, pad_id=0, unk_id=1)


# -------------------------
# 3) Dataset
# -------------------------
class AGNewsDataset(Dataset):
    """
    HuggingFace datasets의 ag_news를 받아서
    (tokenize -> ids -> pad/truncate) 까지 수행하는 Dataset
    """

    def __init__(self, split: str, vocab: Vocab, max_len: int = 128):
        super().__init__()
        self.ds = load_dataset("ag_news", split=split)  # {"text", "label"}
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.ds[idx]
        text = row["text"]
        label = int(row["label"])  # 0~3

        tokens = simple_tokenize(text)
        ids = self.vocab.encode(tokens)

        # truncate
        ids = ids[: self.max_len]

        # padding
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.vocab.pad_id] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)      # (T,)
        label_t = torch.tensor(label, dtype=torch.long)      # ()
        return input_ids, label_t


# -------------------------
# 4) DataLoader helper
# -------------------------
def make_dataloaders(
    max_len: int = 128,
    max_vocab_size: int = 20000,
    min_freq: int = 2,
    batch_size: int = 32,
    num_workers: int = 0,  # mac/cursor 환경에서는 0이 안전
):
    # vocab은 반드시 train 텍스트로만 만들어야 함(데이터 누수 방지)
    train_ds_raw = load_dataset("ag_news", split="train")
    train_texts = [x["text"] for x in train_ds_raw]

    vocab = build_vocab(
        texts=train_texts,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
    )

    train_ds = AGNewsDataset(split="train", vocab=vocab, max_len=max_len)
    test_ds = AGNewsDataset(split="test", vocab=vocab, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return vocab, train_loader, test_loader
