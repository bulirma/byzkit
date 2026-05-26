import numpy as np
import torch
from torch.nn import functional as F

import random


def pad_batch_images(imgs: torch.Tensor, pad_value=0):
    def pad_horizontal(img, w):
        nonlocal pad_value
        total = w - img.size(2)
        l = random.randint(0, total)
        r = total - l
        return F.pad(img, (l, r, 0, 0), mode='constant', value=pad_value)

    max_w = max(map(lambda img: img.size(2), imgs))
    return torch.stack([pad_horizontal(img, max_w) for img in imgs])

def collate(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    lengths = [b[1].size(0) for b in batch]
    images_padded = pad_batch_images(imgs, pad_value=1.0)
    targets = torch.cat(targets).long()
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images_padded, targets, lengths

def symmetric_pad(array: np.ndarray, axis: int, size: int, value):
    total = size - array.shape[axis]
    before = total // 2
    after = total - before
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (before, after)
    return np.pad(array, pad_width, mode='constant', constant_values=value)
