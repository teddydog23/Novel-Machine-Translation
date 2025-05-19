import torch

def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in loader:
            src = src.transpose(0, 1).to(device)
            trg = trg.transpose(0, 1).to(device)

            output = model(src, trg, teacher_forcing_ratio=0.5) 

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].reshape(-1)

            loss = loss_fn(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)