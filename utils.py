import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Tuple


def load_images(path: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    listdir = os.listdir(path)
    if verbose:
        listdir = tqdm(listdir)

    for image_name in listdir:
        img_path = '{}/{}'.format(path, image_name)
        # Catch label from name
        label = int(image_name[-5])

        # Artifact from macOS
        if image_name == '.DS_Store':
            continue

        im = np.array(Image.open(img_path))
        images.append(im)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels
