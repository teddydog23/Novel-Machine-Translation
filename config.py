import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_PATH = r"C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Desktop\NMT1\dataset\train\processed_data.pt"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/model.pt"

# Model hyperparams
EMBED_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.3

# Train settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
N_EPOCHS = 10
CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5
PATIENCE = 5 # Early stopping patience
EVAL_BLEU_EVERY = 5