import torch
from torch.nn import functional as F


def pad_batch_images(imgs: torch.Tensor, pad_value=0):
    def pad(img, h, w):
        nonlocal pad_value
        vt = h - img.size(1)
        t = vt // 2
        b = vt - t
        r = w - img.size(2)
        return F.pad(img, (0, r, t, b), mode='constant', value=pad_value)

    max_h = max(map(lambda img: img.size(1), imgs))
    max_w = max(map(lambda img: img.size(2), imgs))
    return torch.stack([pad(img, max_h, max_w) for img in imgs])

def collate(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    lengths = [b[1].size(0) for b in batch]
    images_padded = pad_batch_images(imgs, pad_value=1.0)
    targets = torch.cat(targets).long()
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images_padded, targets, lengths
