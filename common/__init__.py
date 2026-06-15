import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch

import colorsys
from enum import Enum
import math
import os
from typing import Union, List
import shutil


class Color(Enum):
    BLACK = 0
    RED = 1

class Distrubution:
    def __init__(self, denominator: int, distribution: dict = None):
        self.denom = denominator
        self.dist = dict()
        if distribution is not None:
            self.extend(distribution)

    def remainder(self):
        total = sum(self.dist.values())
        return self.denom - total

    def add(self, value: str, numerator: int):
        if numerator > self.remainder():
            raise ValueError('the distribution sum is greater than 1')
        self.dist[value] = numerator

    def extend(self, distribution: dict):
        for value, numerator in distribution.items():
            self.add(value, numerator)

    def make_uniform(self, values: list):
        remainder = self.remainder()
        for key in values:
            if key in self.dist:
                raise ValueError('the distribution already contains the value')
            self.dist[key] = remainder / self.denom

    def make_cumulative(self) -> dict:
        dist = dict()
        cum_val = 0
        for key in self.dist:
            cum_val += self.dist[key]
            dist[key] = cum_val
        return dist


def get_color(r: int, g: int, b: int) -> Color:
    r, g, b = [c / 255 for c in (r, g, b)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360
    if (h_deg <= 20 or h_deg >= 340) and s > 0.2:
        return Color.RED
    return Color.BLACK

def plt_show(img: Union[cv2.Mat, np.ndarray, torch.Tensor], title: str = None):
    plt.imshow(img, cmap='gray')
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
        ax.imshow(img, cmap='gray')
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

def plt_show_column_grid(
    imgs: List[Union[cv2.Mat, np.ndarray, torch.Tensor]],
    titles: List[str],
    columns: int = 1
):
    a = 1
    padding = 0.2
    n = len(imgs)
    if n == 0:
        return
    cols = columns
    rows = int(np.ceil(n / cols))
    plt_show_mult(imgs, titles, rows, cols, a, padding)

def dec_width(n: int):
    if n == 0:
        return 1
    w = math.ceil(math.log10(n))
    if 10 ** w == n:
        return w + 1
    return w

def is_existing_dir(path: str):
    return os.path.exists(path) and os.path.isdir(path)

def empty_dir(path: str):
    with os.scandir(path) as it:
        for entry in it:
            p = entry.path
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(p)
            elif entry.is_symlink():
                os.unlink(p)
            else:
                os.remove(p)

def levenshtein_distance(a: torch.Tensor, b: torch.Tensor):
    sa = a.size(0)
    sb = b.size(0)
    a = a.tolist()
    b = b.tolist()

    dists = list(range(sb + 1))
    for i in range(1, sa + 1):
        prev_dists = dists
        dists = [i] + [0] * sb
        for j in range(1, sb + 1):
            dists[j] = min(
                dists[j - 1] + 1,
                prev_dists[j] + 1,
                prev_dists[j - 1] + (a[i - 1] != b[j - 1])
            )

    return dists[sb]
