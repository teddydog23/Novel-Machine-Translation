# loss.py
import torch
import torch.nn as nn

def get_loss_function(pad_id):
    """
    Returns CrossEntropyLoss with padding token ignored.
    
    Args:
        pad_id (int): ID of the padding token.
    
    Returns:
        nn.CrossEntropyLoss: Loss function instance.
    """
    return nn.CrossEntropyLoss(ignore_index=pad_id)