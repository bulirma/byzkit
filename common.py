import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch

from typing import Union, List


def plt_show(img: Union[cv2.Mat, np.ndarray, torch.Tensor], title: str = None):
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def plt_show_mult(
    imgs: List[Union[cv2.Mat, np.ndarray, torch.Tensor]],
    titles: List[str],
    rows: int,
    cols: int,
    side: int,
    padding: float
):
    n = len(imgs)
    lt = len(titles)
    if n == 0:
        return
    if n == 1:
        plt_show(imgs[0], None if lt == 0 else titles[0])
        return

    fig, axes = plt.subplots(rows, cols, figsize=(cols * side, rows * side))
    for i, (ax, img) in enumerate(zip(axes.flatten(), imgs)):
        ax.imshow(img)
        ax.axis('off')
        if i < lt and titles[i] is not None:
            ax.set_title(titles[i])

    for ax in axes.flatten()[n:]:
        ax.axis('off')

    plt.tight_layout(pad=padding)
    plt.show()


def plt_show_grid(imgs: List[Union[cv2.Mat, np.ndarray, torch.Tensor]], titles: List[str]):
    a = 1
    ratio = 16/9
    padding = 0.2
    n = len(imgs)
    if n == 0:
        return
    cols = int(np.ceil(np.sqrt(n * ratio)))
    rows = int(np.ceil(n / cols))
    plt_show_mult(imgs, titles, rows, cols, a, padding)


def plt_show_column_grid(imgs: List[Union[cv2.Mat, np.ndarray, torch.Tensor]], titles: List[str], columns: int = 1):
    a = 1
    padding = 0.2
    n = len(imgs)
    if n == 0:
        return
    cols = columns
    rows = int(np.ceil(n / cols))
    plt_show_mult(imgs, titles, rows, cols, a, padding)
