import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import mlflow
import mlflow.pytorch


def train(model, iterator, optimizer, criterion, clip, device, epoch):
    model.train()

    epoch_loss = 0
    for i, batch in tqdm(enumerate(iterator)):

        for key, value in batch.items():
            batch[key] = value.to(device)

        optimizer.zero_grad()

        output = model(batch)
        loss = criterion(output, batch['labels'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    train_loss = epoch_loss / len(iterator)
    mlflow.log_metric('train_loss', train_loss, step=epoch)

    return train_loss


def evaluate(model, iterator, criterion, device, epoch):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(inputs, attention_mask)
            loss = criterion(output, labels)

            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(iterator)
    mlflow.log_metric('val_loss', epoch_loss, step=epoch)
    return epoch_loss
