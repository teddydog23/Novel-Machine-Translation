import torch.optim as optim

def get_optimizer(model, lr, weight_decay=0.0):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_plateau_scheduler(optimizer, factor=0.5, patience=2, verbose=True):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=verbose)