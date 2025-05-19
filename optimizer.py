# optimizer.py
import torch.optim as optim

def get_optimizer_and_scheduler(model):
    """
    Returns AdamW optimizer and ReduceLROnPlateau scheduler.
    
    Args:
        model: PyTorch model whose parameters will be optimized.
    
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    return optimizer, scheduler