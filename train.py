import torch
import numpy as np
from collections import defaultdict
from typing import Any, List, Tuple, DefaultDict
from sklearn.metrics import accuracy_score, classification_report


# train loop on epoch
def train_epoch(model, dataloader, criterion, optimizer, device) -> DefaultDict[str, List[float]]:
    metrics = defaultdict(lambda: list())

    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs).squeeze(-1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            ground_truth = labels.cpu().numpy().astype(np.int64)
            pred = (outputs > 0.).cpu().numpy().astype(np.int64)

        metrics['loss'].append(loss.item())
        metrics['ground_truth'].append(ground_truth)
        metrics['prediction'].append(pred)

    return metrics


# validation loop on epoch
def validate_epoch(model, dataloader, criterion, device):
    metrics = defaultdict(lambda: list())

    model.eval()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)

            ground_truth = labels.cpu().numpy().astype(np.int64)
            pred = (outputs > 0.).cpu().numpy().astype(np.int64)

        metrics['loss'].append(loss.item())
        metrics['ground_truth'].append(ground_truth)
        metrics['prediction'].append(pred)

    return metrics


def metrics_callback(metrics: DefaultDict[str, List[float]]) -> Tuple[float, Any]:
    y_true = np.hstack(metrics['ground_truth'])
    y_pred = np.hstack(metrics['prediction'])

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    report = classification_report(y_true=y_true, y_pred=y_pred)

    return accuracy, report


def train(model, trainloader, valloader, criterion, optimizer, device, n_epochs, verbose):
    for epoch in range(n_epochs):
        if verbose:
            print(f'epoch [{epoch + 1}/{n_epochs}]\n')

        # train
        train_metrics = train_epoch(model, trainloader, criterion, optimizer, device)

        # train metrics
        train_accuracy, train_report = metrics_callback(train_metrics)

        if verbose:
            print(f'train accuracy: {train_accuracy:.4f}\n')
            print(f'train metrics:\n{train_report}\n')

        # validate
        val_metrics = validate_epoch(model, valloader, criterion, device)

        # train metrics
        val_accuracy, val_report = metrics_callback(val_metrics)

        if verbose:
            print(f'val accuracy: {val_accuracy:.4f}\n')
            print(f'val metrics:\n{val_report}\n')
            print(f'{"="*53}\n')
