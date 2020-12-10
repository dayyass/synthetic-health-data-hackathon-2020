import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset


def load_images(path: str):
    images = []
    labels = []
    for image_name in tqdm(os.listdir(path)):
        img_path = '{}/{}'.format(path, image_name)
        # Catch label from name
        label = image_name[-5]

        # Artifact from macOS
        if image_name == '.DS_Store':
            continue

        im = np.array(Image.open(img_path))
        images.append(im)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


class MRIDataset(Dataset):

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Optional[Callable] = None):
        self.images = images
        self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# train loop on epoch
def train(model, dataloader, criterion, optimizer, device):
    
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


# validation loop
def validate(model, dataloader, criterion, device):
    
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
