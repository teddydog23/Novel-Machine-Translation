import torch.nn as nn

# def masked_nll_loss(output, target, pad_idx):
#     """
#     output: [batch*tgt_len, vocab_size] (log_probs)
#     target: [batch*tgt_len]
#     """
#     criterion = nn.NLLLoss(ignore_index=pad_idx)
#     return criterion(output, target)

def get_loss(pad_idx, smoothing=0.1):
    return nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=smoothing)
    
def compute_loss(output, target, loss_fn):
    """
    output: [batch*tgt_len, vocab_size] (logits or log-probs)
    target: [batch*tgt_len]
    """
    return loss_fn(output, target)