import cv2
import lmdb
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import io
import random


class SplitDataset(Dataset):
    def __init__(
        self,
        lmdb_env: lmdb.Environment,
        metadata: dict,
        db_prefix: str,
        transform=None,
        max_height: int = None,
        seed: int = None
    ):
        random.seed(seed)
        super().__init__()
        self.env = lmdb_env
        self.data_db = self.env.open_db(db_prefix.encode() + b'_data')
        self.targets_db = self.env.open_db(db_prefix.encode() + b'_targets')
        self.samples = metadata[db_prefix]['samples']
        self.max_height = metadata['sample_image_max_height']
        self.key_width = metadata[db_prefix]['key_width']
        self.transform = transform
        if max_height is not None and max_height < self.max_height:
            self.max_height = max_height

    def key(self, idx: int) -> bytes:
        return str(idx).zfill(self.key_width).encode()

    def downsize(self, img: np.ndarray):
        h, w = img.shape[:2]
        if h <= self.max_height:
            return img
        ar = self.max_height / h
        width = round(w * ar)
        return cv2.resize(img, (width, self.max_height), interpolation=cv2.INTER_AREA)

    def pad_vertical(self, img: torch.Tensor, value):
        h = self.max_height - img.size(1)
        if h <= 0:
            return img
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

        img = self.downsize(img)

        data = torch.from_numpy(img)
        data = data.permute(2, 0, 1)

        if self.transform is not None:
            data = self.transform(data)
            data = self.pad_vertical(data, 1.0)
        else:
            data = self.pad_vertical(data, 255)

        target_buf = io.BytesIO(target_value)
        target = torch.from_numpy(np.load(target_buf, allow_pickle=False)['target'])

        return data, target
