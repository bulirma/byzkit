import cv2
import lmdb
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import math
import os
import pickle
import random
from typing import Union, List
import shutil


class SplitDataset(Dataset):
    def __init__(self, lmdb_env: lmdb.Environment, db_prefix: str, transform=None, seed: int = None):
        random.seed(seed)
        super().__init__()
        self.env = lmdb_env
        self.data_db = self.env.open_db(db_prefix.encode() + b'_data')
        self.targets_db = self.env.open_db(db_prefix.encode() + b'_targets')
        with self.env.begin(write=False) as txn:
            metadata = pickle.loads(txn.get(b'metadata'))
        self.samples = metadata[db_prefix]['samples']
        self.max_height = metadata['sample_image_max_height']
        self.key_width = metadata[db_prefix]['key_width']
        self.transform = transform

    def key(self, idx: int) -> bytes:
        return str(idx).zfill(self.key_width).encode()

    def pad_vertical(self, img: torch.Tensor, value):
        h = self.max_height - img.size(1)
        t = random.randint(0, h)
        b = h - t
        return F.pad(img, (0, 0, t, b), mode='constant', value=value)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int):
        key = self.key(idx)
        with self.env.begin(write=False) as txn:
            data_value = txn.get(key, db=self.data_db)
            target_value = txn.get(key, db=self.targets_db)
        img = cv2.imdecode(np.frombuffer(data_value, np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = torch.from_numpy(img)
        data = data.permute(2, 0, 1)
        if self.transform is not None:
            data = self.transform(data)
            data = self.pad_vertical(data, 1.0)
        else:
            data = self.pad_vertical(data, 255)
        target = torch.from_numpy(pickle.loads(target_value))
        return data, target


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

def dec_width(n: int):
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

def ser(gold: torch.Tensor, pred: torch.Tensor):
    assert gold.size(0) == pred.size(0)

    ser_err, ser_total = 0, 0
    for i in range(gold.size(0)):
        g = gold[i, :]
        p = pred[i, :]
        ser_err += levenshtein_distance(g, p)
        ser_total += len(g)

    assert ser_total > 0
    return ser_err / ser_total

