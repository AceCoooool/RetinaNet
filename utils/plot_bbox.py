"""Visualize image."""
import random
import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.colors import default_color


def plot_image(img, ax=None, reverse_rgb=False):
    if ax is None:
        # create new axes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = img.copy()
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    ax.imshow(img.astype(np.uint8))
    return ax


def plot_bbox(img, pred, class_names=None, colors=default_color,
              reverse_rgb=False, ax=None):
    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    boxes = pred.bbox
    scores = pred.get_field("scores").tolist()
    labels = pred.get_field("labels").tolist()

    if len(boxes) < 1:
        return ax

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(boxes):
        cls_id = int(labels[i]) - 1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=2.5)
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores[i]) if scores is not None else ''
        if class_name or score:
            ax.text(xmin, ymin - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=11, color='white')
    return ax
