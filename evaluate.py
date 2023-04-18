import torch
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_accuracy
import wandb


def evaluate(model, device, dataloader, number_of_classes, criterion):
    model.eval()

    running_loss = 0.0
    predictions, targets = [], []

    for i, (inputs, labels) in enumerate(dataloader):
        if i%10 == 0:
            print(f'{i}-th sample - evaluation')
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            loss = criterion(outputs, labels) 

        running_loss += loss.item() * inputs.size(0)
        predictions.append(outputs.cpu())
        targets.append(labels.cpu())
    predictions = torch.cat(predictions, 0)
    targets = torch.cat(targets, 0)

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = dice(predictions, targets)
    epoch_acc = binary_accuracy(predictions, targets)

    if wandb.run is not None:
        wandb.log({'val_loss': epoch_loss, 'val_dice': epoch_dice, 'val_acc': epoch_acc})

    return epoch_loss, epoch_dice, epoch_acc