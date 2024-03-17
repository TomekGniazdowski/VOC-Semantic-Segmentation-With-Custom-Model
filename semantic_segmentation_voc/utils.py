import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

from semantic_segmentation_voc.transforms import ColorDenormalization


def display_images(
    dataset: list, 
    rows: int = 5,
    size: int = 7
    ):
    
    img_ids = np.random.randint(0, len(dataset), rows)
    dataset = [dataset[i] for i in img_ids]
    images = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    color_map = generate_color_map(labels)
    _, axs = plt.subplots(rows, 3, figsize=(5, size))
    axs[0, 0].set_title("Images")
    axs[0, 1].set_title("Images unnormalized") 
    axs[0, 2].set_title("Labels")
    for i in range(rows):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img_unnormalized = ColorDenormalization(images[i]).squeeze().permute(1, 2, 0).cpu().numpy()
        label = labels[i].cpu().numpy()
        label_3d = generate_color_mask(label, color_map)
        axs[i, 0].imshow(np.clip(img, 0, 1))
        axs[i, 1].imshow(img_unnormalized)
        axs[i, 2].imshow(label_3d)
    plt.show()
    
def display_batch_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    images: torch.Tensor,
    size: int 
    ):
    color_map = generate_color_map(labels)
    _, axs = plt.subplots(len(predictions), 3, figsize=(5, size))
    axs[0, 0].set_title("Predictions")
    axs[0, 1].set_title("Labels")
    axs[0, 2].set_title("Images")
    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        prediction = torch.argmax(prediction, dim=0).squeeze().cpu().numpy()
        label = label.cpu().numpy()
        prediction_mask = generate_color_mask(prediction, color_map)
        label_mask = generate_color_mask(label, color_map)
        axs[i, 0].imshow(prediction_mask)
        axs[i, 1].imshow(label_mask)
        axs[i, 2].imshow(np.clip(images[i].permute(1, 2, 0).cpu().numpy(), 0, 1))
    plt.show()

def generate_color_mask(x: np.array, color_map: dict):
    mask_3d = np.zeros((x.shape[0], x.shape[1], 3))
    mask_3d[x == 0] = [0, 0, 0]
    mask_3d[x == 255] = [255, 255, 255]
    for k, v in color_map.items():
        mask_3d[x == k] = v
    mask_3d = mask_3d.astype(np.uint8)
    return mask_3d

def generate_color_map(x: torch.Tensor):
    classes = get_classes(x)
    random_colors = torch.randint(0, 256, (len(classes), 3)).tolist()
    color_map = dict(zip(classes, random_colors))
    return color_map

def get_classes(x: torch.Tensor):
    all_classes = []
    for x_ in tqdm(x):
        classes = torch.unique(x_).tolist()
        if 0 in classes:
            classes.remove(0)
        if 255 in classes:
            classes.remove(255)
        all_classes += classes
    return list(set(all_classes))

def error_analysis_per_image(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    device: str
):
    error_analysis = []
    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        loss_val = loss_fn(prediction.unsqueeze(dim=0).to(device), 
                            label.unsqueeze(dim=0).to(device)).sum() # batch dim
        prediction = torch.argmax(prediction, dim=0).squeeze()
        error_analysis.append((i, prediction.cpu(), label.cpu(), loss_val.item()))
    return pd.DataFrame(error_analysis, columns=['index', 'prediction', 'label', 'loss'])

def display_error_analysis(
    error_per_img_analysis: pd.DataFrame,
    size: int
):
    _, axs = plt.subplots(len(error_per_img_analysis), 2, figsize=(5, size))
    color_map = generate_color_map([r['label'] for _, r in error_per_img_analysis.iterrows()])
    axs[0, 0].set_title("Predictions")
    axs[0, 1].set_title("Labels")
    for i, (_, r) in enumerate(error_per_img_analysis.iterrows()):
        axs[i, 0].imshow(generate_color_mask(r['prediction'].numpy(), color_map))
        axs[i, 1].imshow(generate_color_mask(r['label'].numpy(), color_map))
    plt.show()

def plot_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor
):
    all_predictions, all_labels = [], []
    for prediction, label in zip(predictions, labels):
        prediction = torch.argmax(prediction, dim=0).cpu().numpy().flatten()
        label = label.cpu().numpy().flatten()
        all_predictions.extend(prediction)
        all_labels.extend(label)
    cm = confusion_matrix(all_labels, all_predictions, normalize='true')
    cm = np.round(cm, 2)
    cm[cm == 0] = np.nan
    _, ax = plt.subplots(figsize=(15, 15))
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    sns.heatmap(cm, annot=True, ax=ax, cmap="YlGnBu")