import os
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from PIL import Image
from tqdm import tqdm


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_images(path: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    listdir = os.listdir(path)
    if verbose:
        listdir = tqdm(listdir)

    for image_name in listdir:
        img_path = "{}/{}".format(path, image_name)
        # Catch label from name
        label = int(image_name[-5])

        # Artifact from macOS
        if image_name == ".DS_Store":
            continue

        im = np.array(Image.open(img_path))
        images.append(im)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def shap_deep_explainer(
    model: torch.nn.Module,
    background: torch.Tensor,
    test_images: torch.Tensor,
):
    """
    https://shap.readthedocs.io/en/latest/example_notebooks/deep_explainer/PyTorch%20Deep%20Explainer%20MNIST%20example.html
    """
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

    # plot the feature attributions
    shap.image_plot(shap_numpy, test_numpy)


def visualize_scatter(
    data_2d: np.ndarray,
    labels_idx: np.ndarray,
    idx2label: Dict[int, str],
    title: str = "",
    figsize: Tuple = (10, 10),
) -> None:
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.grid()

    for label_id in np.unique(labels_idx):
        plt.scatter(
            data_2d[np.where(labels_idx == label_id), 0],
            data_2d[np.where(labels_idx == label_id), 1],
            label=idx2label[label_id],
        )

    plt.legend()
