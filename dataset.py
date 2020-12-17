import numpy as np
from typing import Optional, Callable
from torch.utils.data import Dataset


class MRIDataset(Dataset):

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Optional[Callable] = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# TRANSFORMS

class Unsqueeze:

    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, img):
        return np.expand_dims(img, axis=self.axis)


class Repeat:

    def __init__(self, n_channel, axis=1):
        self.n_channel = n_channel
        self.axis = axis

    def __call__(self, img):
        return np.repeat(img, repeats=self.n_channel, axis=self.axis)
