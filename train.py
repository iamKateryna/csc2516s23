import torch
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_accuracy
import wandb


def train(model, device, dataloader, number_of_classes, criterion, optimizer):
    model.train()

    running_loss = 0.0
    predictions, targets = [], []

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(labels, outputs)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predictions.append(outputs.cpu())
        targets.append(labels.cpu())
    predictions = torch.cat(predictions, 0)
    targets = torch.cat(targets, 0)

    # Calculate epoch loss and metrics
    epoch_dice = dice(predictions, targets)#, num_classes=number_of_classes, multi_class=True)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = binary_accuracy(predictions, targets)

    if wandb.run is not None:
        wandb.log({'train_loss': epoch_loss, 'train_dice': epoch_dice, 'train_acc': epoch_acc})

    return epoch_loss, epoch_dice, epoch_acc