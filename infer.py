import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from seq2seq_attention.encoder import Encoder
from seq2seq_attention.decoder import Decoder
from seq2seq_attention.seq2seq import Seq2Seq
from dataset.dataloader import TranslationDataset, collate_fn
from config import *
import nltk
nltk.download('wordnet')

# === Decode function for BLEU/METEOR
def decode_sequence(seq, idx2word):
    return [idx2word.get(idx, "<UNK>") for idx in seq if idx2word.get(idx) not in ["<PAD>", "<EOS>", "<SOS>"]]

def calculate_bleu(preds, refs):
    refs = [[r] for r in refs]  # wrap reference
    return corpus_bleu(refs, preds, smoothing_function=SmoothingFunction().method1)

def calculate_meteor(preds, refs):
    scores = [meteor_score([ref], pred) for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores)

def evaluate_model(model_path, split="val"):
    print(f"🔍 Evaluating model: {model_path} on split: {split}")

    data = torch.load(DATA_PATH)
    src_lines = data["src_lines"][split]
    tgt_lines = data["tgt_lines"][split]
    vocab_en = data["vocab_en"]
    vocab_vi = data["vocab_vi"]
    pad_idx = data["pad_idx"]

    idx2vi = {v: k for k, v in vocab_vi.items()}

    dataset = TranslationDataset(src_lines, tgt_lines, vocab_en, vocab_vi)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, pad_idx))

    INPUT_DIM = len(vocab_en)
    OUTPUT_DIM = len(vocab_vi)

    encoder = Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    decoder = Decoder(EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    model = Seq2Seq(encoder, decoder).to(DEVICE)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds, refs = [], []

    with torch.no_grad():
        for src, trg in tqdm(loader, desc="🚀 Evaluating"):
            src = src.transpose(0, 1).to(DEVICE)
            trg = trg.transpose(0, 1).to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            pred_ids = output.argmax(-1).transpose(0, 1).tolist()
            ref_ids = trg.transpose(0, 1).tolist()

            for pred_seq, ref_seq in zip(pred_ids, ref_ids):
                pred = decode_sequence(pred_seq, idx2vi)
                ref = decode_sequence(ref_seq, idx2vi)
                preds.append(pred)
                refs.append(ref)

    bleu = calculate_bleu(preds, refs)
    meteor = calculate_meteor(preds, refs)
    print(f"\n📏 BLEU Score: {bleu:.4f}")
    print(f"🪐 METEOR Score: {meteor:.4f}")

    # Optional: Print 5 example translations
    print("\n📚 Example Translations:")
    for i in range(5):
        print(f"🔹 Source:  {src_lines[i]}")
        print(f"🔸 Target:  {tgt_lines[i]}")
        print(f"✅ Predicted: {' '.join(preds[i])}")
        print("-" * 50)

if __name__ == "__main__":
    # Example usage:
    evaluate_model("checkpoints/model.pt", split="val")
