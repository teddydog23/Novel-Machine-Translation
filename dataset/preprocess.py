import os
import re
import torch
import random
import nltk
from pyvi import ViTokenizer
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# 1. Tokenizers
def tokenize_en(text):
    return text.lower().strip().split()

def tokenize_vi(text):
    return ViTokenizer.tokenize(text.lower()).split()

# 2. Vocab builder
def build_vocab(tokenized_sentences, min_freq=2, specials=["<PAD>", "<SOS>", "<EOS>", "<UNK>"]):
    counter = Counter()
    for tokens in tokenized_sentences:
        counter.update(tokens)
    vocab = {token: idx for idx, token in enumerate(specials)}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

# 3. Encode sentence
def encode(tokens, vocab, sos="<SOS>", eos="<EOS>"):
    ids = [vocab.get(sos)]
    ids += [vocab.get(t, vocab.get("<UNK>")) for t in tokens]
    ids.append(vocab.get(eos))
    return ids

# 4. PyTorch Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, src_vocab, tgt_vocab):
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_tokens = tokenize_en(self.src_lines[idx])
        tgt_tokens = tokenize_vi(self.tgt_lines[idx])
        src_ids = torch.tensor(encode(src_tokens, self.src_vocab), dtype=torch.long)
        tgt_ids = torch.tensor(encode(tgt_tokens, self.tgt_vocab), dtype=torch.long)
        return src_ids, tgt_ids

# 5. Collate function
def collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=True)
    return src_batch, tgt_batch

# 6. Main function
def preprocess_and_save(en_path, vi_path, save_path, batch_size=32, min_freq=2,
                        train_ratio=0.8, val_ratio=0.1, seed=42):
    print("Loading raw data...")
    with open(en_path, encoding="utf-8") as f:
        en_lines = [line.strip() for line in f if line.strip()]
    with open(vi_path, encoding="utf-8") as f:
        vi_lines = [line.strip() for line in f if line.strip()]

    assert len(en_lines) == len(vi_lines), "Số lượng câu không khớp!"

    MAX_LEN = 80

    filtered_en = []
    filtered_vi = []

    for en, vi in zip(en_lines, vi_lines):
        if len(en.split()) <= MAX_LEN and len(vi.split()) <= MAX_LEN:
            filtered_en.append(en)
            filtered_vi.append(vi)

    en_lines = filtered_en[:500000]
    vi_lines = filtered_vi[:500000]

    print("Tokenizing...")
    tokenized_en = [tokenize_en(line) for line in en_lines]
    tokenized_vi = [tokenize_vi(line) for line in vi_lines]

    print("Building vocabularies...")
    vocab_en = build_vocab(tokenized_en, min_freq)
    vocab_vi = build_vocab(tokenized_vi, min_freq)

    pad_idx = vocab_en["<PAD>"]

    print("Creating full dataset...")
    full_dataset = TranslationDataset(en_lines, vi_lines, vocab_en, vocab_vi)

    # Chia tập train - val - test
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")

    # Đảm bảo chia reproducible
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"💾 Saving to {save_path}")
    torch.save({
        "src_lines": {
            "train": [en_lines[i] for i in train_dataset.indices],
            "val": [en_lines[i] for i in val_dataset.indices],
            "test": [en_lines[i] for i in test_dataset.indices],
        },
        "tgt_lines": {
            "train": [vi_lines[i] for i in train_dataset.indices],
            "val": [vi_lines[i] for i in val_dataset.indices],
            "test": [vi_lines[i] for i in test_dataset.indices],
        },
        "vocab_en": vocab_en,
        "vocab_vi": vocab_vi,
        "pad_idx": pad_idx,
        "params": {
            "en_path": en_path,
            "vi_path": vi_path,
            "min_freq": min_freq,
            "batch_size": batch_size,
            "ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "test": round(1 - train_ratio - val_ratio, 2)
            },
            "seed": seed
        }
    }, save_path)

    print("Preprocessing & splitting done!")

# 🔧 Gọi hàm
if __name__ == "__main__":
    preprocess_and_save(
        en_path=r"C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Desktop\NMT1\dataset\train\train.en",
        vi_path=r"C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Desktop\NMT1\dataset\train\train.vi",
        save_path=r"C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Desktop\NMT1\dataset\train\processed_data.pt",
        batch_size=32,
        min_freq=2
    )