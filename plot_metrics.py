import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_CSV_PATH = r"D:\NMT1\logs\train_metrics.csv"
SAVE_DIR = "logs"

def plot_metrics():
    df = pd.read_csv(LOG_CSV_PATH)

    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    bleu = pd.to_numeric(df["bleu"], errors='coerce')

    # === Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Val Loss", marker="x")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.show()

    # === Plot BLEU Score
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, bleu, label="BLEU Score", color="green", marker="s")
    plt.title("BLEU Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, "bleu_curve.png"))
    plt.show()

if __name__ == "__main__":
    plot_metrics()